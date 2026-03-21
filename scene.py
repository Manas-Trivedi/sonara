"""
Scene analysis: fuse depth map + object detections into a ranked list of
nearby objects. This is the single source of truth for the data contract
consumed by both the video overlay and the audio/haptic JS component.

DATA CONTRACT (each object in the returned list):
{
    "label":      str,    # YOLO class name, e.g. "person"
    "proximity":  float,  # 0.0 (far, ~5m+) -> 1.0 (very close, <0.2m)
    "x_pos":      float,  # 0.0 (far left) -> 1.0 (far right)
    "distance_m": float,  # derived human-readable distance in meters
    "bbox":       [x1, y1, x2, y2],  # pixel coords in the *processed* frame
    "danger":     bool    # proximity > DANGER_THRESHOLD
}
The list is sorted by proximity descending and truncated to TOP_K.
"""

from __future__ import annotations
import numpy as np

TOP_K = 3
DANGER_THRESHOLD = 0.75
MAX_DISTANCE_M = 5.0
MIN_DISTANCE_M = 0.2


def proximity_to_meters(proximity: float) -> float:
    """Linear inverse mapping. Clamped to [MIN, MAX]."""
    d = (1.0 - proximity) * MAX_DISTANCE_M
    return float(np.clip(d, MIN_DISTANCE_M, MAX_DISTANCE_M))


def analyze(
    depth_map: np.ndarray,
    detections: list[dict],
    frame_w: int,
    frame_h: int,
) -> list[dict]:
    """
    Args:
        depth_map: (H, W) float32, normalized 0-1, 1 = NEAR (MiDaS inverse depth).
                   Must be same resolution as the frame detections were run on.
        detections: list of {"label": str, "conf": float, "bbox": (x1,y1,x2,y2)}
                    bbox in pixel coords matching depth_map dimensions.
        frame_w, frame_h: dimensions of the processed frame.

    Returns:
        Top-K objects sorted by proximity (closest first). May be fewer than K.
    """
    if depth_map is None or len(detections) == 0:
        return []

    dh, dw = depth_map.shape[:2]
    results = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Clamp bbox to depth map bounds and ensure non-empty
        x1c = int(np.clip(x1, 0, dw - 1))
        y1c = int(np.clip(y1, 0, dh - 1))
        x2c = int(np.clip(x2, x1c + 1, dw))
        y2c = int(np.clip(y2, y1c + 1, dh))

        region = depth_map[y1c:y2c, x1c:x2c]
        if region.size == 0:
            continue

        # Use a high percentile instead of plain mean so that a small close
        # object inside a large box (e.g. a pole against far background)
        # still registers as close. Cheap and robust.
        proximity = float(np.percentile(region, 85))
        proximity = float(np.clip(proximity, 0.0, 1.0))

        cx = (x1 + x2) / 2.0
        x_pos = float(np.clip(cx / frame_w, 0.0, 1.0))

        results.append({
            "label": det["label"],
            "proximity": proximity,
            "x_pos": x_pos,
            "distance_m": proximity_to_meters(proximity),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "danger": proximity > DANGER_THRESHOLD,
        })

    results.sort(key=lambda o: o["proximity"], reverse=True)
    return results[:TOP_K]


# ---------------------------------------------------------------------------
# Mock-data smoke test — run `python scene.py` before wiring anything else.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    H, W = 288, 384

    # Synthetic depth: left half far (0.1), right half near (0.9),
    # with a very-near blob in the top-right corner.
    depth = np.full((H, W), 0.1, dtype=np.float32)
    depth[:, W // 2:] = 0.9
    depth[0:80, W - 100:W] = 0.98

    mock_dets = [
        {"label": "person", "conf": 0.92, "bbox": (10, 50, 120, 250)},      # left / far
        {"label": "chair",  "conf": 0.80, "bbox": (200, 100, 300, 260)},    # right / near
        {"label": "bottle", "conf": 0.70, "bbox": (W - 90, 10, W - 10, 70)},# top-right / very near
        {"label": "tv",     "conf": 0.60, "bbox": (150, 150, 190, 190)},    # mid / should rank last
    ]

    out = analyze(depth, mock_dets, W, H)
    print(json.dumps(out, indent=2))

    assert len(out) == 3, "Expected top-3 truncation"
    assert out[0]["label"] == "bottle", f"Closest should be bottle, got {out[0]['label']}"
    assert out[0]["danger"] is True, "Bottle should be in danger zone"
    assert out[0]["x_pos"] > 0.7, "Bottle should be on the right"
    assert out[-1]["proximity"] < out[0]["proximity"], "List must be sorted desc"
    print("\n✓ scene.py mock test passed")
