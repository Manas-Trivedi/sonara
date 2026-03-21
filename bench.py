"""
Standalone pipeline benchmark — no camera required.
Runs the exact same stages as Processor._loop with synthetic frames
and prints per-stage timings. Use this to diagnose bottlenecks without
needing the phone stream.

    python bench.py
"""
from __future__ import annotations
import time
import json
import os
from pathlib import Path

import cv2
import numpy as np

import scene
from depth import DepthEstimator
from detection import Detector

PROC_W, PROC_H = 384, 288
N_FRAMES = 60
STATIC_DIR = Path(__file__).parent / "static"
SCENE_JSON = STATIC_DIR / "scene.json"
STATIC_DIR.mkdir(exist_ok=True)


def make_frame(i: int) -> np.ndarray:
    """Synthetic 640x480 frame with some structure so models do real work."""
    f = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(f, (100 + i % 50, 100), (300 + i % 50, 400), (200, 180, 60), -1)
    cv2.circle(f, (450, 200 + i % 30), 80, (60, 60, 200), -1)
    return f


def render_annotated(frame_bgr, depth_map, objects):
    out = frame_bgr.copy()
    if depth_map is not None:
        heat = (depth_map * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)
        out = cv2.addWeighted(out, 0.6, heat, 0.4, 0)
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return out


def write_scene_json(objects):
    tmp = SCENE_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"objects": objects, "ts": time.time()}, f)
    os.replace(tmp, SCENE_JSON)


def run(depth_model, detector, label):
    print(f"\n=== {label} ===")
    print(f"Depth: {depth_model.info()}  |  Detect: {detector.info()}")

    acc = {"grab": 0.0, "resize": 0.0, "depth": 0.0, "yolo": 0.0,
           "scene": 0.0, "annotate": 0.0, "write": 0.0, "total": 0.0}
    last_depth = None

    # warmup
    wf = cv2.resize(make_frame(0), (PROC_W, PROC_H))
    for _ in range(3):
        depth_model.estimate(wf)
        detector.detect(wf)

    DEPTH_EVERY, YOLO_EVERY = 3, 2
    last_dets: list[dict] = []

    for i in range(N_FRAMES):
        t0 = time.perf_counter()
        raw = make_frame(i)  # simulates .latest()
        t1 = time.perf_counter()

        frame = cv2.resize(raw, (PROC_W, PROC_H))
        t2 = time.perf_counter()

        if i % DEPTH_EVERY == 0 or last_depth is None:
            last_depth = depth_model.estimate(frame)
        depth_map = last_depth
        t3 = time.perf_counter()

        if i % YOLO_EVERY == 0:
            last_dets = detector.detect(frame)
        detections = last_dets
        t4 = time.perf_counter()

        objects = scene.analyze(depth_map, detections, PROC_W, PROC_H)
        t5 = time.perf_counter()

        write_scene_json(objects)
        t6 = time.perf_counter()

        ann = render_annotated(frame, depth_map, objects)
        _, _buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
        t7 = time.perf_counter()

        acc["grab"]     += (t1 - t0) * 1000
        acc["resize"]   += (t2 - t1) * 1000
        acc["depth"]    += (t3 - t2) * 1000
        acc["yolo"]     += (t4 - t3) * 1000
        acc["scene"]    += (t5 - t4) * 1000
        acc["write"]    += (t6 - t5) * 1000
        acc["annotate"] += (t7 - t6) * 1000
        acc["total"]    += (t7 - t0) * 1000

    n = N_FRAMES
    tot = acc["total"] / n
    print(
        f"[PERF] grab={acc['grab']/n:.1f}ms  resize={acc['resize']/n:.1f}ms  "
        f"depth={acc['depth']/n:.1f}ms  yolo={acc['yolo']/n:.1f}ms  "
        f"scene={acc['scene']/n:.1f}ms  annotate={acc['annotate']/n:.1f}ms  "
        f"write={acc['write']/n:.1f}ms  total={tot:.1f}ms → {1000/tot:.1f} FPS"
    )
    return {k: v / n for k, v in acc.items()}


if __name__ == "__main__":
    print("Loading models (baseline config)…")
    dm = DepthEstimator()
    det = Detector()
    run(dm, det, "BASELINE")
