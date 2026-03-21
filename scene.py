"""
Scene analysis: depth-map-first obstacle detection.

The depth map is divided into a 3x2 grid (left/center/right x upper/lower).
Each cell's interior ROI is sampled and converted into a proximity reading. YOLO
detections are optional context — if a detection's center falls inside a
cell, its label is attached. If YOLO sees nothing, the grid still fires.

DATA CONTRACT (each zone dict in the returned list):
{
    "zone":       str,    # "left" | "center" | "right"
    "level":      str,    # "upper" | "lower"
    "proximity":  float,  # 0.0 (far) -> 1.0 (very close)
    "distance_m": float,  # (1 - proximity) * 5, clamped 0.2–5.0
    "label":      str|None,  # YOLO class if a detection overlaps, else None
    "bbox":       [x1,y1,x2,y2],  # grid-cell pixel bounds (for overlay)
    "danger":     bool    # proximity > DANGER_THRESHOLD
}
Always returns exactly 6 zones (all cells), sorted by proximity descending.
"""

from __future__ import annotations
import numpy as np

DANGER_THRESHOLD = 0.75
MAX_DISTANCE_M = 5.0
MIN_DISTANCE_M = 0.2

COLS = ("left", "center", "right")
ROWS = ("upper", "lower")
INNER_X_MARGIN_RATIO = 0.12
UPPER_Y_MARGIN_RATIO = 0.10
LOWER_TOP_KEEP_RATIO = 0.62
LOWER_BOTTOM_MARGIN_RATIO = 0.08


def _clean_label(label) -> str | None:
    if label is None:
        return None

    cleaned = str(label).strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null"}:
        return None
    return cleaned


def proximity_to_meters(proximity: float) -> float:
    d = (1.0 - proximity) * MAX_DISTANCE_M
    return float(np.clip(d, MIN_DISTANCE_M, MAX_DISTANCE_M))


def _sample_cell_for_proximity(cell: np.ndarray, level: str) -> np.ndarray:
    """
    Sample an interior ROI instead of the full cell.

    The lower row is trimmed more aggressively so floor/ground pixels at the
    bottom of the frame do not constantly register as near obstacles.
    """
    h, w = cell.shape[:2]
    x_margin = max(1, int(w * INNER_X_MARGIN_RATIO))
    y_margin = max(1, int(h * UPPER_Y_MARGIN_RATIO))

    x1 = min(x_margin, max(0, w - 1))
    x2 = max(x1 + 1, w - x_margin)

    if level == "lower":
        y1 = min(y_margin, max(0, h - 1))
        y2 = max(y1 + 1, int(h * LOWER_TOP_KEEP_RATIO))
        y2 = min(y2, max(y1 + 1, h - max(1, int(h * LOWER_BOTTOM_MARGIN_RATIO))))
    else:
        y1 = min(y_margin, max(0, h - 1))
        y2 = max(y1 + 1, h - y_margin)

    sample = cell[y1:y2, x1:x2]
    return sample if sample.size else cell


def analyze(
    depth_map: np.ndarray,
    detections: list[dict],
    frame_w: int,
    frame_h: int,
) -> list[dict]:
    """
    Args:
        depth_map: (H, W) float32, normalized 0-1, 1 = NEAR (MiDaS inverse depth).
        detections: list of {"label": str, "conf": float, "bbox": (x1,y1,x2,y2)}.
                    May be empty — grid still produces output.
        frame_w, frame_h: dimensions of the processed frame.

    Returns:
        6 zone dicts sorted by proximity descending.
    """
    if depth_map is None:
        return []

    dh, dw = depth_map.shape[:2]
    col_edges = [0, dw // 3, 2 * dw // 3, dw]
    row_edges = [0, dh // 2, dh]

    zones = []
    for ri, level in enumerate(ROWS):
        y1, y2 = row_edges[ri], row_edges[ri + 1]
        for ci, zone in enumerate(COLS):
            x1, x2 = col_edges[ci], col_edges[ci + 1]

            cell = depth_map[y1:y2, x1:x2]
            sample = _sample_cell_for_proximity(cell, level)
            proximity = float(np.clip(np.percentile(sample, 90), 0.0, 1.0))

            zones.append({
                "zone": zone,
                "level": level,
                "proximity": proximity,
                "distance_m": proximity_to_meters(proximity),
                "label": None,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "danger": proximity > DANGER_THRESHOLD,
                "_conf": -1.0,  # internal: best YOLO conf seen in this cell
            })

    # Attach YOLO labels: detection center → containing cell.
    for det in detections:
        label = _clean_label(det.get("label"))
        if label is None:
            continue

        bx1, by1, bx2, by2 = det["bbox"]
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0

        ci = 0 if cx < col_edges[1] else 1 if cx < col_edges[2] else 2
        ri = 0 if cy < row_edges[1] else 1
        z = zones[ri * 3 + ci]

        if det["conf"] > z["_conf"]:
            z["_conf"] = det["conf"]
            z["label"] = label

    for z in zones:
        del z["_conf"]

    zones.sort(key=lambda z: z["proximity"], reverse=True)
    return zones


# ---------------------------------------------------------------------------
# Mock-data smoke test — run `python scene.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    H, W = 288, 384

    # Synthetic depth: mostly far (0.1), with a near wall on the right (0.85)
    # and a very-near blob in upper-right.
    depth = np.full((H, W), 0.1, dtype=np.float32)
    depth[:, 2 * W // 3:] = 0.85
    depth[0:80, W - 100:W] = 0.97

    mock_dets = [
        {"label": "person", "conf": 0.92, "bbox": (10, 50, 120, 250)},      # left
        {"label": "chair",  "conf": 0.80, "bbox": (W - 90, 160, W - 10, 270)},  # lower-right
        {"label": "bottle", "conf": 0.40, "bbox": (W - 80, 170, W - 20, 260)},  # lower-right, lower conf
    ]

    out = analyze(depth, mock_dets, W, H)
    print(json.dumps(out, indent=2))

    assert len(out) == 6, f"Expected 6 zones, got {len(out)}"
    assert out[0]["zone"] == "right" and out[0]["level"] == "upper", \
        f"Closest should be upper-right, got {out[0]['zone']}/{out[0]['level']}"
    assert out[0]["danger"] is True, "Upper-right should be in danger zone"
    assert out[0]["label"] is None, "Upper-right has no YOLO detection"

    # lower-right should have the chair label (higher conf than bottle)
    lower_right = next(z for z in out if z["zone"] == "right" and z["level"] == "lower")
    assert lower_right["label"] == "chair", f"Expected chair, got {lower_right['label']}"
    assert lower_right["proximity"] > 0.3, "Lower-right wall should register"

    # Grid still fires with zero detections
    out_empty = analyze(depth, [], W, H)
    assert len(out_empty) == 6, "Grid must produce 6 zones even with no detections"
    assert out_empty[0]["proximity"] > 0.3, "Wall must still be detected without YOLO"

    print("\n✓ scene.py mock test passed")
