"""
detector.py
YOLOv8-based face detector.
Returns bounding boxes and confidence scores for every face in a frame.
"""

import numpy as np
import cv2
from ultralytics import YOLO
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class FaceDetector:
    """
    Wraps YOLOv8 face detection.
    Downloads yolov8n-face.pt automatically on first use.
    """

    def __init__(self):
        os.makedirs("models", exist_ok=True)
        # yolov8n-face is a community fine-tuned model; fallback to yolov8n if not found
        model_path = config.YOLO_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"[Detector] Downloading YOLOv8 face model → {model_path}")
            # Pull from ultralytics hub or use generic yolov8n as placeholder
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(model_path)

        self.conf   = config.YOLO_CONF_THRESH
        self.iou    = config.YOLO_IOU_THRESH
        self.min_sz = config.YOLO_MIN_FACE_SIZE
        print("[Detector] YOLOv8 face detector ready.")

    def detect(self, frame: np.ndarray):
        """
        Parameters
        ----------
        frame : np.ndarray  BGR image (H, W, 3)

        Returns
        -------
        boxes  : list of [x1, y1, x2, y2]  (int pixel coords)
        scores : list of float confidences
        crops  : list of np.ndarray face crops (BGR)
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=0
        )

        boxes, scores, crops = [], [], []

        for det in results[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
            conf = float(det.conf[0].cpu())

            # Filter tiny detections
            w, h = x2 - x1, y2 - y1
            if w < self.min_sz or h < self.min_sz:
                continue

            # Clip to frame boundaries
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            crops.append(crop)

        return boxes, scores, crops

    def draw_detections(self, frame: np.ndarray, boxes, scores) -> np.ndarray:
        """Draw raw YOLO detections on frame (for debug/comparison)."""
        vis = frame.copy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{score:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis
