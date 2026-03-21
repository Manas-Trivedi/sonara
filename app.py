"""
Sonara — real-time spatial-audio navigation aid.
Streamlit entry point.

Run:
    streamlit run app.py --server.address=0.0.0.0
Then open http://<laptop-ip>:8501 on the PHONE browser so vibration works.
"""

from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import scene
from depth import DepthEstimator
from detection import Detector

# ---------------------------------------------------------------------------
PROC_W, PROC_H = 384, 288
STATIC_DIR = Path(__file__).parent / "static"
SCENE_JSON = STATIC_DIR / "scene.json"
STATIC_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Sonara", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Model loading (cached once per process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading MiDaS depth model…")
def load_depth():
    return DepthEstimator()


@st.cache_resource(show_spinner="Loading YOLOv8n…")
def load_detector():
    return Detector()


# ---------------------------------------------------------------------------
# Background frame grabber (decoupled from render loop)
# ---------------------------------------------------------------------------
class FrameGrabber:
    """Continuously pulls the latest frame from an MJPEG stream in a daemon
    thread. Only the newest frame is kept; stale frames are dropped."""

    def __init__(self):
        self._url = None
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self, url: str):
        self.stop()
        self._url = url
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None

    def _loop(self):
        self._cap = cv2.VideoCapture(self._url)
        while not self._stop.is_set():
            ok, f = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = f

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


@st.cache_resource
def get_grabber():
    return FrameGrabber()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_annotated(frame_bgr, depth_map, objects):
    """Overlay depth heatmap (40% opacity) + white bboxes + labels."""
    out = frame_bgr.copy()

    if depth_map is not None:
        heat = (depth_map * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)
        out = cv2.addWeighted(out, 0.6, heat, 0.4, 0)

    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"{o['label']} {o['distance_m']:.1f}m"
        cv2.putText(out, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if o["danger"]:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def render_radar(objects):
    """Return HTML of 3 horizontal proximity bars."""
    rows = []
    for i in range(3):
        if i < len(objects):
            o = objects[i]
            p = o["proximity"]
            # green → yellow → red
            if p > 0.75:
                color = "#e53935"
            elif p > 0.45:
                color = "#fdd835"
            else:
                color = "#43a047"
            pct = int(p * 100)
            label = f"{o['label']} — {o['distance_m']:.1f} m"
        else:
            color, pct, label = "#333", 0, "—"
        rows.append(
            f'<div style="margin:4px 0;font:12px sans-serif;color:#ccc">'
            f'{label}'
            f'<div style="background:#222;height:16px;border-radius:3px">'
            f'<div style="width:{pct}%;height:100%;background:{color};'
            f'border-radius:3px"></div></div></div>'
        )
    return "".join(rows)


def write_scene_json(objects):
    """Atomic write so the JS poller never reads a half-written file."""
    tmp = SCENE_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"objects": objects, "ts": time.time()}, f)
    os.replace(tmp, SCENE_JSON)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
depth_model = load_depth()
detector = load_detector()
grabber = get_grabber()

with st.sidebar:
    st.title("Sonara")
    url = st.text_input("IP Webcam URL",
                        value="http://192.168.1.100:8080/video",
                        help="MJPEG stream from the IP Webcam app")
    running = st.toggle("▶ Run", value=False)
    st.divider()
    st.caption(f"Depth: {depth_model.info()}")
    st.caption(f"Detect: {detector.info()}")
    fps_slot = st.empty()
    st.divider()
    st.caption("Open this page on your **phone** browser and tap "
               "**Enable Audio** below so vibration works.")

# Audio/haptic component — identical HTML every rerun so the iframe persists.
_audio_html = (Path(__file__).parent / "audio_component.html").read_text()
components.html(_audio_html, height=80)

video_slot = st.empty()
radar_slot = st.empty()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if "frame_i" not in st.session_state:
    st.session_state.frame_i = 0
    st.session_state.last_depth = None
    st.session_state.t_prev = time.time()

if running:
    if grabber._thread is None or not grabber._thread.is_alive():
        grabber.start(url)

    raw = grabber.latest()
    if raw is None:
        video_slot.info("Waiting for camera stream…")
        time.sleep(0.2)
        st.rerun()

    frame = cv2.resize(raw, (PROC_W, PROC_H))

    # Depth: skip every other frame, reuse previous map
    if st.session_state.frame_i % 2 == 0 or st.session_state.last_depth is None:
        depth_map = depth_model.estimate(frame)
        st.session_state.last_depth = depth_map
    else:
        depth_map = st.session_state.last_depth

    detections = detector.detect(frame)
    objects = scene.analyze(depth_map, detections, PROC_W, PROC_H)

    write_scene_json(objects)

    annotated = render_annotated(frame, depth_map, objects)
    video_slot.image(annotated, use_container_width=True)
    radar_slot.markdown(render_radar(objects), unsafe_allow_html=True)

    # FPS
    now = time.time()
    dt = now - st.session_state.t_prev
    st.session_state.t_prev = now
    fps = 1.0 / dt if dt > 0 else 0.0
    fps_slot.metric("FPS", f"{fps:.1f}")

    st.session_state.frame_i += 1
    time.sleep(0.05)
    st.rerun()

else:
    grabber.stop()
    write_scene_json([])  # silence audio
    video_slot.info("Press ▶ Run in the sidebar to start.")
    radar_slot.markdown(render_radar([]), unsafe_allow_html=True)
    fps_slot.metric("FPS", "—")
