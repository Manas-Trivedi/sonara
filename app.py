"""
Sonara — real-time spatial-audio navigation aid.
Streamlit entry point.

Run:
    streamlit run app.py --server.address=0.0.0.0
Then open http://<laptop-ip>:8501 on the PHONE browser so vibration works.

Video is served as MJPEG on port 8502 so the browser pulls frames directly
without Streamlit reruns → no flicker.
"""

from __future__ import annotations
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
MJPEG_PORT = 8502
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
# Shared annotated-frame buffer (written by Processor, read by MJPEG server)
# ---------------------------------------------------------------------------
class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self.objects: list[dict] = []
        self.fps: float = 0.0
        # black placeholder so /stream never blocks before first frame
        black = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", black)
        self._placeholder = buf.tobytes()

    def set(self, frame_bgr: np.ndarray, objects: list[dict], fps: float):
        _, buf = cv2.imencode(".jpg", frame_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._lock:
            self._jpeg = buf.tobytes()
            self.objects = objects
            self.fps = fps

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg if self._jpeg is not None else self._placeholder


@st.cache_resource
def get_buffer():
    return FrameBuffer()


# ---------------------------------------------------------------------------
# Processing thread: grab → depth → detect → annotate → buffer + scene.json
# ---------------------------------------------------------------------------
class Processor:
    def __init__(self, grabber, depth_model, detector, buffer):
        self.grabber = grabber
        self.depth_model = depth_model
        self.detector = detector
        self.buffer = buffer
        self.running = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running.clear()

    def _loop(self):
        frame_i = 0
        last_depth = None
        t_prev = time.time()

        while self.running.is_set():
            raw = self.grabber.latest()
            if raw is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(raw, (PROC_W, PROC_H))

            if frame_i % 2 == 0 or last_depth is None:
                last_depth = self.depth_model.estimate(frame)
            depth_map = last_depth

            detections = self.detector.detect(frame)
            objects = scene.analyze(depth_map, detections, PROC_W, PROC_H)

            write_scene_json(objects)

            annotated = render_annotated(frame, depth_map, objects)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = 1.0 / dt if dt > 0 else 0.0

            self.buffer.set(annotated, objects, fps)
            frame_i += 1


@st.cache_resource
def get_processor(_grabber, _depth, _detector, _buffer):
    return Processor(_grabber, _depth, _detector, _buffer)


# ---------------------------------------------------------------------------
# MJPEG HTTP server (port 8502) — started exactly once per process
# ---------------------------------------------------------------------------
def _make_handler(buffer: FrameBuffer):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw):
            pass  # silence

        def do_GET(self):
            if self.path.split("?")[0] != "/stream":
                self.send_error(404)
                return
            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                while True:
                    jpg = buffer.get_jpeg()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.04)  # ~25 fps cap on the wire
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return MJPEGHandler


@st.cache_resource
def start_mjpeg_server(_buffer):
    """Returns True once the server thread is running. Cached so it only
    ever fires once no matter how many times Streamlit reruns."""
    # Allow fast restart of the app without "address already in use"
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", MJPEG_PORT),
                                  _make_handler(_buffer))
    except OSError as e:
        st.warning(f"MJPEG port {MJPEG_PORT} unavailable: {e}")
        return False
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return True


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_annotated(frame_bgr, depth_map, objects):
    """Overlay depth heatmap (40% opacity) + white bboxes + labels.
    Returns BGR for direct JPEG encoding."""
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

    return out


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
buffer = get_buffer()
processor = get_processor(grabber, depth_model, detector, buffer)
start_mjpeg_server(buffer)

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

# Video surface — persistent <img> pulling MJPEG from port 8502.
# Hostname resolved client-side so it works from laptop *and* phone.
components.html(
    f"""
    <img id="sonara-video"
         style="width:100%;border-radius:8px;background:#000;" />
    <script>
      const videoEl = document.getElementById('sonara-video');
      const parentHost = window.parent && window.parent.location
        ? window.parent.location.hostname
        : '';
      const host = parentHost || window.location.hostname || 'localhost';
      const proto = window.parent && window.parent.location
        ? window.parent.location.protocol
        : window.location.protocol || 'http:';
      videoEl.src = proto + '//' + host + ':{MJPEG_PORT}/stream';
    </script>
    """,
    height=320,
)

radar_slot = st.empty()

# ---------------------------------------------------------------------------
# Control loop — only updates radar + FPS (~2 Hz). Video is independent.
# ---------------------------------------------------------------------------
if running:
    if grabber._thread is None or not grabber._thread.is_alive():
        grabber.start(url)
    processor.start()

    radar_slot.markdown(render_radar(buffer.objects), unsafe_allow_html=True)
    fps_slot.metric("FPS", f"{buffer.fps:.1f}")

    time.sleep(0.5)
    st.rerun()

else:
    processor.stop()
    grabber.stop()
    write_scene_json([])  # silence audio
    radar_slot.markdown(render_radar([]), unsafe_allow_html=True)
    fps_slot.metric("FPS", "—")
