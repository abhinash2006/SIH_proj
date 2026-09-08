import math
from typing import List, Dict, Any

class MultiFrameObjectTracker:
    """
    Tracks detected objects (people, vehicles, structures) across consecutive drone frames
    using spatial IoU and Euclidean distance matching to maintain persistent object IDs.
    """

    def __init__(self, max_distance: float = 60.0, iou_thresh: float = 0.2):
        self.max_distance = max_distance
        self.iou_thresh = iou_thresh
        self.class_counters = {}
        self.tracked_objects = [] # Stores active tracks

    def _get_next_id(self, class_name: str) -> str:
        prefix = class_name.upper()
        count = self.class_counters.get(prefix, 0) + 1
        self.class_counters[prefix] = count
        return f"{prefix}_{count:03d}"

    def _compute_iou(self, boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[0] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[0] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def track_frame_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Assigns persistent object IDs to 2D detections in current frame.
        """
        assigned_detections = []

        for det in detections:
            cls_name = det["class_name"]
            cx, cy = det["pixel_center"]
            bbox = det["bbox"]

            matched_id = None
            best_match_score = -1.0

            # Match against existing tracks of same class
            for track in self.tracked_objects:
                if track["class_name"] == cls_name:
                    last_cx, last_cy = track["last_center"]
                    dist = math.hypot(cx - last_cx, cy - last_cy)
                    iou = self._compute_iou(bbox, track["last_bbox"])

                    if iou > self.iou_thresh or dist < self.max_distance:
                        match_score = iou + (1.0 / (1.0 + dist))
                        if match_score > best_match_score:
                            best_match_score = match_score
                            matched_id = track["object_id"]

            if matched_id is None:
                matched_id = self._get_next_id(cls_name)
                self.tracked_objects.append({
                    "object_id": matched_id,
                    "class_name": cls_name,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "last_center": (cx, cy),
                    "last_bbox": bbox,
                    "occurrences": 1
                })
            else:
                # Update track
                for track in self.tracked_objects:
                    if track["object_id"] == matched_id:
                        track["last_frame"] = frame_idx
                        track["last_center"] = (cx, cy)
                        track["last_bbox"] = bbox
                        track["occurrences"] += 1
                        break

            det_copy = det.copy()
            det_copy["object_id"] = matched_id
            assigned_detections.append(det_copy)

        return assigned_detections
