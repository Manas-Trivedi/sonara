"""
YOLOv8n object detection wrapper.

Returns detections in the format expected by scene.analyze():
    [{"label": str, "conf": float, "bbox": (x1, y1, x2, y2)}, ...]
Coordinates are in the *input frame's* pixel space.
"""

from __future__ import annotations
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, conf_threshold: float = 0.35):
        self.model = YOLO("yolov8n.pt")  # auto-downloads on first run
        self.conf_threshold = conf_threshold
        self.names = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        # imgsz left at default; frame is already pre-resized upstream
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            verbose=False,
        )
        out = []
        r = results[0]
        if r.boxes is None:
            return out

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
            out.append({
                "label": self.names[cls],
                "conf": float(conf),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })
        return out

    def info(self) -> str:
        return "YOLOv8n"
