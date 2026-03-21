"""
YOLOv8n object detection wrapper.

Returns detections in the format expected by scene.analyze():
    [{"label": str, "conf": float, "bbox": (x1, y1, x2, y2)}, ...]
Coordinates are in the *input frame's* pixel space.
"""

from __future__ import annotations
import numpy as np
from ultralytics import YOLO


def _normalize_label(raw_label) -> str | None:
    label = str(raw_label).strip()
    if not label:
        return None
    if label.lower() in {"none", "null"}:
        return None
    return label


class Detector:
    def __init__(self, conf_threshold: float = 0.35):
        self.model = YOLO("yolov8n.pt")  # auto-downloads on first run
        self.conf_threshold = conf_threshold
        self.names = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        # Zone-level detection only needs ~320px — YOLO rescales bboxes
        # back to input dimensions automatically.
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            imgsz=320,
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
            label = _normalize_label(self.names[cls])
            if label is None:
                continue

            out.append({
                "label": label,
                "conf": float(conf),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })
        return out

    def info(self) -> str:
        return "YOLOv8n"
