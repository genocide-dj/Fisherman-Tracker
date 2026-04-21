"""
tracker.py
DeepSORT multi-person tracker.
Assigns persistent track IDs across frames using Kalman filter + appearance features.
"""

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class FaceTracker:
    """
    Wraps deep_sort_realtime.DeepSort for face tracking.
    Returns confirmed tracks with stable IDs across occlusions.
    """

    def __init__(self):
        self.tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            nms_max_overlap=config.DEEPSORT_NMS_MAX_OVERLAP,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DIST,
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",          # built-in appearance embedder
            half=False,
            bgr=True,
            embedder_gpu=False,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self._frame_count = 0
        print("[Tracker] DeepSORT tracker ready.")

    def update(self, frame: np.ndarray, boxes, scores):
        """
        Parameters
        ----------
        frame  : np.ndarray  BGR frame
        boxes  : list of [x1, y1, x2, y2]
        scores : list of float

        Returns
        -------
        tracks : list of dict
            {
              'track_id': int,
              'bbox':     [x1, y1, x2, y2],
              'score':    float,
              'is_new':   bool
            }
        """
        self._frame_count += 1

        if len(boxes) == 0:
            self.tracker.update_tracks([], frame=frame)
            return []

        # DeepSORT expects detections as ([x,y,w,h], confidence, class_name)
        detections = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], score, "face"))

        raw_tracks = self.tracker.update_tracks(detections, frame=frame)

        active = []
        for t in raw_tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            active.append({
                "track_id": t.track_id,
                "bbox":     [x1, y1, x2, y2],
                "score":    t.det_conf if t.det_conf else 0.0,
                "age":      t.age,
                "is_new":   t.age <= config.DEEPSORT_N_INIT + 1
            })

        return active

    def draw_tracks(self, frame: np.ndarray, tracks, person_id_map: dict = None) -> np.ndarray:
        """
        Annotate frame with track bounding boxes and IDs.
        person_id_map maps track_id → re-identified person label (optional).
        """
        vis = frame.copy()
        colors = _get_color_palette(20)

        for t in tracks:
            tid  = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            label = person_id_map.get(tid, f"T{tid}") if person_id_map else f"T{tid}"
            color = colors[tid % len(colors)]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis


def _get_color_palette(n: int):
    """Generate n visually distinct BGR colors."""
    import colorsys
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors
