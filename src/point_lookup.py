import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Union

logger = logging.getLogger("PointLookup3D")


class PointLookupResult(dict):
    """
    Dictionary return type that also supports tuple unpacking:
    pos_3d, meta = RobustPointLookup.lookup_3d_point(...)
    """
    def __iter__(self):
        if self.get("location_status") == "LOCALIZED_3D" and self.get("x_m") is not None:
            pt = np.array([self["x_m"], self["y_m"], self["z_m"]], dtype=np.float32)
        else:
            pt = None
        return iter((pt, self))


class RobustPointLookup:
    """
    Robust 3D point lookup engine for 2D object detections in VGGT depth maps and point maps.
    Performs hierarchical multi-scale spatial neighborhood search (center -> 3x3 -> 5x5 -> 9x9 -> 15x15),
    rejecting NaNs, Infs, zero/negative depths, and extreme outliers.
    
    CRITICAL SAFETY RULES:
    1. NEVER returns (0, 0, 0) as fallback coordinates.
    2. NEVER fabricates coordinates.
    3. Returns status="UNAVAILABLE" / None when no valid geometry is found.
    """

    @classmethod
    def lookup_3d_point(
        cls,
        *args,
        point_map: Optional[np.ndarray] = None,
        u_vggt: Optional[float] = None,
        v_vggt: Optional[float] = None,
        u_scaled: Optional[float] = None,
        v_scaled: Optional[float] = None,
        depth_map: Optional[np.ndarray] = None,
        extrinsic: Optional[np.ndarray] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsic: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        bbox_scaled: Optional[Tuple[float, float, float, float]] = None,
        min_depth: float = 0.20,
        max_depth: float = 250.0
    ) -> PointLookupResult:
        """
        Looks up the 3D metric world coordinate (X, Y, Z).
        Supports:
        - lookup_3d_point(point_map, u_vggt, v_vggt, ...)
        - lookup_3d_point(u_scaled, v_scaled, depth_map, extrinsic, intrinsic, ...)
        - named parameters
        """
        ext = extrinsic if extrinsic is not None else extrinsics
        intr = intrinsic if intrinsic is not None else intrinsics

        # Parse positional args
        p_map = point_map
        u_val = u_vggt if u_vggt is not None else u_scaled
        v_val = v_vggt if v_vggt is not None else v_scaled
        d_map = depth_map

        if len(args) > 0:
            if isinstance(args[0], np.ndarray) and args[0].ndim == 3 and args[0].shape[-1] == 3:
                # Format: lookup_3d_point(point_map, u_vggt, v_vggt, ...)
                p_map = args[0]
                if len(args) > 1:
                    u_val = float(args[1])
                if len(args) > 2:
                    v_val = float(args[2])
            elif len(args) >= 5 and isinstance(args[2], np.ndarray):
                # Format: lookup_3d_point(u_scaled, v_scaled, depth_map, extrinsic, intrinsic)
                u_val = float(args[0])
                v_val = float(args[1])
                d_map = args[2]
                ext = args[3]
                intr = args[4]

        # Case 1: Direct 3D Point Map Sampling (VGGT Point Cloud / Point Head)
        if p_map is not None and p_map.size > 0 and u_val is not None and v_val is not None:
            H, W = p_map.shape[:2]
            u_int = int(round(u_val))
            v_int = int(round(v_val))

            if 0 <= u_int < W and 0 <= v_int < H:
                # Direct check
                pt_direct = p_map[v_int, u_int]
                if np.isfinite(pt_direct).all() and (np.linalg.norm(pt_direct) > 1e-3) and (np.linalg.norm(pt_direct) < 500.0):
                    return PointLookupResult({
                        "location_status": "LOCALIZED_3D",
                        "status": "DIRECT_HIT",
                        "x_m": round(float(pt_direct[0]), 3),
                        "y_m": round(float(pt_direct[1]), 3),
                        "z_m": round(float(pt_direct[2]), 3),
                        "sampled_depth_m": round(float(pt_direct[2]), 3),
                        "neighborhood_size": 1,
                        "reason": "Direct valid 3D point from point map."
                    })

                # Neighborhood multi-scale search (radius 1, 2, 4, 7)
                window_half_sizes = [1, 2, 4, 7]
                for r in window_half_sizes:
                    y1 = max(0, v_int - r)
                    y2 = min(H, v_int + r + 1)
                    x1 = max(0, u_int - r)
                    x2 = min(W, u_int + r + 1)
                    patch = p_map[y1:y2, x1:x2].reshape(-1, 3)
                    valid_mask = (
                        np.isfinite(patch).all(axis=1) &
                        (np.linalg.norm(patch, axis=1) > 1e-3) &
                        (np.linalg.norm(patch, axis=1) < 500.0)
                    )
                    cand = patch[valid_mask]
                    if len(cand) > 0:
                        med_pt = np.median(cand, axis=0)
                        return PointLookupResult({
                            "location_status": "LOCALIZED_3D",
                            "status": f"NEIGHBORHOOD_{r*2+1}x{r*2+1}",
                            "x_m": round(float(med_pt[0]), 3),
                            "y_m": round(float(med_pt[1]), 3),
                            "z_m": round(float(med_pt[2]), 3),
                            "sampled_depth_m": round(float(med_pt[2]), 3),
                            "neighborhood_size": r * 2 + 1,
                            "reason": f"Localized via {len(cand)} neighborhood points in {r*2+1}x{r*2+1} window."
                        })

            # If point map didn't have valid points and no depth map, return FAILED
            if d_map is None or ext is None or intr is None:
                return PointLookupResult({
                    "location_status": "UNAVAILABLE",
                    "status": "FAILED",
                    "x_m": None, "y_m": None, "z_m": None,
                    "sampled_depth_m": None,
                    "neighborhood_size": 0,
                    "reason": "No valid geometry found in point map."
                })

        # Case 2: Depth Map Unprojection using camera intrinsics and extrinsics
        if d_map is None or d_map.size == 0 or ext is None or intr is None or u_val is None or v_val is None:
            return PointLookupResult({
                "location_status": "UNAVAILABLE",
                "status": "FAILED",
                "x_m": None, "y_m": None, "z_m": None,
                "sampled_depth_m": None,
                "neighborhood_size": 0,
                "reason": "Missing depth map or camera geometry parameters."
            })

        if d_map.ndim == 3:
            d_map = d_map.squeeze(-1)

        H, W = d_map.shape[:2]
        u_int = int(round(u_val))
        v_int = int(round(v_val))

        if u_int < 0 or u_int >= W or v_int < 0 or v_int >= H:
            return PointLookupResult({
                "location_status": "UNAVAILABLE",
                "status": "FAILED",
                "x_m": None, "y_m": None, "z_m": None,
                "sampled_depth_m": None,
                "neighborhood_size": 0,
                "reason": f"Center pixel ({u_int}, {v_int}) out of depth map bounds ({W}x{H})."
            })

        window_half_sizes = [0, 1, 2, 4, 7]
        valid_depths = np.array([], dtype=np.float32)
        used_radius = 0

        for r in window_half_sizes:
            y1 = max(0, v_int - r)
            y2 = min(H, v_int + r + 1)
            x1 = max(0, u_int - r)
            x2 = min(W, u_int + r + 1)

            if bbox_scaled is not None:
                bx1, by1, bx2, by2 = bbox_scaled
                x1 = max(x1, int(bx1))
                x2 = min(x2, int(bx2) + 1)
                y1 = max(y1, int(by1))
                y2 = min(y2, int(by2) + 1)
                if x2 <= x1 or y2 <= y1:
                    continue

            patch = d_map[y1:y2, x1:x2]
            candidates = patch[
                np.isfinite(patch) &
                (patch >= min_depth) &
                (patch <= max_depth) &
                (patch > 1e-3)
            ]

            if len(candidates) > 0:
                valid_depths = candidates
                used_radius = r * 2 + 1 if r > 0 else 1
                break

        if len(valid_depths) == 0:
            return PointLookupResult({
                "location_status": "UNAVAILABLE",
                "status": "FAILED",
                "x_m": None, "y_m": None, "z_m": None,
                "sampled_depth_m": None,
                "neighborhood_size": 0,
                "reason": "No valid depth in local neighborhood."
            })

        sampled_depth = float(np.median(valid_depths))
        fu, fv = float(intr[0, 0]), float(intr[1, 1])
        cu, cv = float(intr[0, 2]), float(intr[1, 2])

        x_c = (u_val - cu) * sampled_depth / fu
        y_c = (v_val - cv) * sampled_depth / fv
        z_c = sampled_depth

        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        pt_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        pt_world = R_c2w @ pt_cam + t_c2w
        X, Y, Z = float(pt_world[0]), float(pt_world[1]), float(pt_world[2])

        return PointLookupResult({
            "location_status": "LOCALIZED_3D",
            "status": "DIRECT_HIT" if used_radius == 1 else f"NEIGHBORHOOD_{used_radius}x{used_radius}",
            "x_m": round(X, 3),
            "y_m": round(Y, 3),
            "z_m": round(Z, 3),
            "sampled_depth_m": round(sampled_depth, 3),
            "neighborhood_size": used_radius,
            "reason": f"Localized via depth map (window: {used_radius}x{used_radius})."
        })
