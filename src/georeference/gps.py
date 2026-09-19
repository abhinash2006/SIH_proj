"""
GPS Telemetry Parser and Local Geodetic Projector.
Parses, validates, and projects WGS84 coordinates (lat, lon, alt) to metric Cartesian ENU/UTM.
"""

import os
import csv
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0          # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2


@dataclass
class GPSPoint:
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    x_enu: float
    y_enu: float
    z_enu: float


@dataclass
class GPSTrajectory:
    points: List[GPSPoint]
    ref_latitude: float
    ref_longitude: float
    ref_altitude: float
    total_distance_m: float
    timestamps: np.ndarray
    positions_enu: np.ndarray  # [N, 3] in meters (East, North, Up)


class GPSTelemetryParser:
    """
    Parses and validates UAV flight GPS CSV logs.
    Converts geographic coordinates into local Cartesian East-North-Up (ENU) coordinates.
    """

    def __init__(
        self,
        max_speed_mps: float = 35.0,  # Max realistic drone speed for outlier detection (35 m/s ~ 126 km/h)
        ref_lat: Optional[float] = None,
        ref_lon: Optional[float] = None,
        ref_alt: Optional[float] = None,
        coordinate_system: str = "ENU",
    ):
        self.max_speed_mps = max_speed_mps
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.coordinate_system = coordinate_system.upper()

    @staticmethod
    def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
        """Converts WGS84 (lat, lon, alt) to Earth-Centered, Earth-Fixed (ECEF) Cartesian (X, Y, Z)."""
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        x = (N + alt_m) * cos_lat * cos_lon
        y = (N + alt_m) * cos_lat * sin_lon
        z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
        return float(x), float(y), float(z)

    @classmethod
    def geodetic_to_enu(
        cls,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
    ) -> Tuple[float, float, float]:
        """
        Converts WGS84 coordinates to local East-North-Up (ENU) Cartesian meters
        relative to an origin point (ref_lat, ref_lon, ref_alt).
        """
        x, y, z = cls.geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        rx, ry, rz = cls.geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

        dx = x - rx
        dy = y - ry
        dz = z - rz

        lat_r = np.radians(ref_lat)
        lon_r = np.radians(ref_lon)
        sin_lat = np.sin(lat_r)
        cos_lat = np.cos(lat_r)
        sin_lon = np.sin(lon_r)
        cos_lon = np.cos(lon_r)

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return float(e), float(n), float(u)

    def parse_csv(self, csv_file_path: str) -> GPSTrajectory:
        """
        Reads CSV file, validates rows, removes duplicates & jumps,
        and projects to local ENU coordinates.
        """
        csv_p = Path(csv_file_path)
        if not csv_p.exists():
            raise FileNotFoundError(f"GPS telemetry CSV file not found: {csv_file_path}")

        raw_points = []
        with open(csv_p, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Find matching column names (case-insensitive)
            field_map = {}
            for field in reader.fieldnames or []:
                fn = field.strip().lower()
                if "time" in fn or "ts" in fn:
                    field_map["timestamp"] = field
                elif "lat" in fn:
                    field_map["latitude"] = field
                elif "lon" in fn or "lng" in fn:
                    field_map["longitude"] = field
                elif "alt" in fn or "height" in fn:
                    field_map["altitude"] = field

            for row_idx, row in enumerate(reader):
                try:
                    ts_val = float(row.get(field_map.get("timestamp", "timestamp"), row_idx))
                    lat_val = float(row[field_map["latitude"]])
                    lon_val = float(row[field_map["longitude"]])
                    alt_val = float(row.get(field_map.get("altitude", "altitude"), 0.0))

                    # 1. Geographic bounds check
                    if not (-90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0):
                        logger.warning(f"[GPS] Row {row_idx}: Coordinate out of bounds ({lat_val}, {lon_val}), skipping.")
                        continue

                    # 2. Altitude sanity check
                    if not (-500.0 <= alt_val <= 9000.0):
                        logger.warning(f"[GPS] Row {row_idx}: Unrealistic altitude ({alt_val}m), skipping.")
                        continue

                    raw_points.append((ts_val, lat_val, lon_val, alt_val))
                except (KeyError, ValueError) as e:
                    continue

        if not raw_points:
            raise ValueError(f"No valid GPS entries could be parsed from {csv_file_path}")

        # Sort by timestamp
        raw_points.sort(key=lambda p: p[0])

        # Remove duplicate timestamps and velocity jump outliers
        filtered_points = [raw_points[0]]
        for i in range(1, len(raw_points)):
            prev_ts, prev_lat, prev_lon, prev_alt = filtered_points[-1]
            curr_ts, curr_lat, curr_lon, curr_alt = raw_points[i]

            dt = curr_ts - prev_ts
            if dt <= 1e-4:
                # Duplicate timestamp, ignore
                continue

            # Check approximate distance jump
            # 1 deg lat ~ 111,139 meters
            dlat_m = (curr_lat - prev_lat) * 111139.0
            dlon_m = (curr_lon - prev_lon) * 111139.0 * np.cos(np.radians(prev_lat))
            dalt_m = curr_alt - prev_alt
            dist_m = np.sqrt(dlat_m**2 + dlon_m**2 + dalt_m**2)

            speed_mps = dist_m / dt
            if speed_mps > self.max_speed_mps:
                logger.warning(
                    f"[GPS] High velocity jump detected ({speed_mps:.1f} m/s > {self.max_speed_mps} m/s) at t={curr_ts:.1f}s. Skipping outlier."
                )
                continue

            filtered_points.append(raw_points[i])

        # Establish reference anchor (first valid GPS coordinate if not specified)
        ref_lat = self.ref_lat if self.ref_lat is not None else filtered_points[0][1]
        ref_lon = self.ref_lon if self.ref_lon is not None else filtered_points[0][2]
        ref_alt = self.ref_alt if self.ref_alt is not None else filtered_points[0][3]

        gps_point_list: List[GPSPoint] = []
        positions_enu = np.zeros((len(filtered_points), 3), dtype=np.float64)
        timestamps = np.zeros(len(filtered_points), dtype=np.float64)
        total_dist = 0.0

        for idx, (ts, lat, lon, alt) in enumerate(filtered_points):
            e, n, u = self.geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
            positions_enu[idx] = [e, n, u]
            timestamps[idx] = ts

            if idx > 0:
                step_dist = np.linalg.norm(positions_enu[idx] - positions_enu[idx - 1])
                total_dist += step_dist

            gps_point_list.append(
                GPSPoint(
                    timestamp=ts,
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    x_enu=e,
                    y_enu=n,
                    z_enu=u,
                )
            )

        logger.info(
            f"[GPS] Loaded {len(gps_point_list)} GPS points | "
            f"Anchor: ({ref_lat:.6f}, {ref_lon:.6f}, {ref_alt:.1f}m) | Trajectory length: {total_dist:.1f}m"
        )

        return GPSTrajectory(
            points=gps_point_list,
            ref_latitude=ref_lat,
            ref_longitude=ref_lon,
            ref_altitude=ref_alt,
            total_distance_m=round(total_dist, 2),
            timestamps=timestamps,
            positions_enu=positions_enu,
        )


def parse_gps_csv(csv_path: str, **kwargs) -> GPSTrajectory:
    parser = GPSTelemetryParser(**kwargs)
    return parser.parse_csv(csv_path)
