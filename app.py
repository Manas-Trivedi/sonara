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
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
# Background frame source manager (IP Webcam or browser camera upload)
# ---------------------------------------------------------------------------
class VideoSource:
    """Provides the latest frame from either an MJPEG URL or browser uploads."""

    def __init__(self):
        self._mode = "ip_webcam"
        self._url = None
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def use_ip_webcam(self, url: str):
        restart = self._mode != "ip_webcam" or self._url != url
        self._mode = "ip_webcam"
        if not restart and self._thread and self._thread.is_alive():
            return

        self._stop_reader()
        self.clear_frame()
        self._url = url
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def use_browser_camera(self):
        self._mode = "browser_camera"
        self._stop_reader()
        self.clear_frame()

    def stop(self):
        self._mode = "stopped"
        self._url = None
        self._stop_reader()
        self.clear_frame()

    def clear_frame(self):
        with self._lock:
            self._frame = None

    def set_browser_frame(self, jpg_bytes: bytes) -> bool:
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._lock:
            self._frame = frame
        return True

    def _stop_reader(self):
        self._stop.set()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None

    def _reader_loop(self):
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
def get_video_source():
    return VideoSource()


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
    def __init__(self, video_source, depth_model, detector, buffer):
        self.video_source = video_source
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

    DEPTH_EVERY = 3   # run MiDaS on 1 of every 3 frames
    YOLO_EVERY = 2    # run YOLO on 1 of every 2 frames

    def _loop(self):
        frame_i = 0
        last_depth = None
        last_dets: list[dict] = []
        t_prev = time.time()

        # --- perf instrumentation ---
        perf_acc = {"grab": 0.0, "resize": 0.0, "depth": 0.0, "yolo": 0.0,
                    "scene": 0.0, "annotate": 0.0, "write": 0.0, "total": 0.0}
        PERF_EVERY = 30

        while self.running.is_set():
            t0 = time.perf_counter()
            raw = self.video_source.latest()
            t1 = time.perf_counter()
            if raw is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(raw, (PROC_W, PROC_H))
            t2 = time.perf_counter()

            if frame_i % self.DEPTH_EVERY == 0 or last_depth is None:
                last_depth = self.depth_model.estimate(frame)
            depth_map = last_depth
            t3 = time.perf_counter()

            if frame_i % self.YOLO_EVERY == 0:
                last_dets = self.detector.detect(frame)
            detections = last_dets
            t4 = time.perf_counter()

            objects = scene.analyze(depth_map, detections, PROC_W, PROC_H)
            t5 = time.perf_counter()

            write_scene_json(objects)
            t6 = time.perf_counter()

            annotated = render_annotated(frame, depth_map, objects)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = 1.0 / dt if dt > 0 else 0.0

            self.buffer.set(annotated, objects, fps)
            t7 = time.perf_counter()

            perf_acc["grab"]     += (t1 - t0) * 1000
            perf_acc["resize"]   += (t2 - t1) * 1000
            perf_acc["depth"]    += (t3 - t2) * 1000
            perf_acc["yolo"]     += (t4 - t3) * 1000
            perf_acc["scene"]    += (t5 - t4) * 1000
            perf_acc["write"]    += (t6 - t5) * 1000
            perf_acc["annotate"] += (t7 - t6) * 1000
            perf_acc["total"]    += (t7 - t0) * 1000

            frame_i += 1
            if frame_i % PERF_EVERY == 0:
                n = PERF_EVERY
                avg_total = perf_acc["total"] / n
                print(
                    f"[PERF] grab={perf_acc['grab']/n:.0f}ms  "
                    f"resize={perf_acc['resize']/n:.0f}ms  "
                    f"depth={perf_acc['depth']/n:.0f}ms  "
                    f"yolo={perf_acc['yolo']/n:.0f}ms  "
                    f"scene={perf_acc['scene']/n:.0f}ms  "
                    f"annotate={perf_acc['annotate']/n:.0f}ms  "
                    f"write={perf_acc['write']/n:.0f}ms  "
                    f"total={avg_total:.0f}ms → {1000/avg_total:.1f} FPS",
                    flush=True,
                )
                for k in perf_acc:
                    perf_acc[k] = 0.0


@st.cache_resource
def get_processor(_video_source, _depth, _detector, _buffer):
    return Processor(_video_source, _depth, _detector, _buffer)


# ---------------------------------------------------------------------------
# MJPEG/ingest server (port 8502) — started exactly once per process
# ---------------------------------------------------------------------------
def _make_handler(buffer: FrameBuffer, video_source: VideoSource):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw):
            pass  # silence

        def _send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

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
                self._send_cors_headers()
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

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/upload_frame":
                self.send_error(404)
                return

            query = parse_qs(parsed.query)
            mode = query.get("mode", [""])[0]
            if mode != "browser_camera":
                self.send_response(409)
                self._send_cors_headers()
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            ok = video_source.set_browser_frame(body)
            self.send_response(200 if ok else 400)
            self._send_cors_headers()
            self.end_headers()

    return MJPEGHandler


@st.cache_resource
def start_mjpeg_server(_buffer, _video_source):
    """Returns True once the server thread is running. Cached so it only
    ever fires once no matter how many times Streamlit reruns."""
    # Allow fast restart of the app without "address already in use"
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", MJPEG_PORT),
                                  _make_handler(_buffer, _video_source))
    except OSError as e:
        st.warning(f"MJPEG port {MJPEG_PORT} unavailable: {e}")
        return False

    cert_file = st.get_option("server.sslCertFile")
    key_file = st.get_option("server.sslKeyFile")
    if cert_file and key_file:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)

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
        label_name = o["label"] if o.get("label") not in (None, "", "None", "none") else f"{o['zone']}-{o['level']}"
        label = f"{label_name} {o['distance_m']:.1f}m"
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
            label_name = o["label"] if o.get("label") not in (None, "", "None", "none") else f"{o['zone']}-{o['level']}"
            label = f"{label_name} — {o['distance_m']:.1f} m"
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
video_source = get_video_source()
buffer = get_buffer()
processor = get_processor(video_source, depth_model, detector, buffer)
start_mjpeg_server(buffer, video_source)

with st.sidebar:
    st.title("Sonara")
    source_mode = st.radio(
        "Video Source",
        ["IP Webcam feed", "This device camera"],
    )
    source_key = "ip_webcam" if source_mode == "IP Webcam feed" else "browser_camera"

    url = st.text_input("IP Webcam URL",
                        value="http://192.168.1.100:8080/video",
                        help="MJPEG stream from the IP Webcam app")
    running = st.toggle("Start Camera Processing", value=False)
    st.caption(
        "Turn on **Start Camera Processing** to begin the selected camera feed."
    )
    st.divider()
    st.caption(f"Depth: {depth_model.info()}")
    st.caption(f"Detect: {detector.info()}")
    fps_slot = st.empty()
    st.divider()
    st.caption("Open this page on your **phone** browser and tap "
               "**Enable Audio** below so vibration works.")

if source_key == "browser_camera":
    st.warning(
        "Device camera mode needs camera permission in the phone browser, and "
        "most phones only allow that on HTTPS or localhost. If you opened "
        "Sonara as plain `http://<laptop-ip>:8501`, the browser may keep the "
        "feed blank without ever prompting. In that case, either run Streamlit "
        "over HTTPS or use IP Webcam mode."
    )

st.caption(
    "Use **This device camera** when you open Sonara directly on your phone. "
    "Then switch on **Start Camera Processing** in the left sidebar. "
    "The processed video below still comes from the laptop after inference."
)

# Audio/haptic component — identical HTML every rerun so the iframe persists.
_audio_html = (Path(__file__).parent / "audio_component.html").read_text()
components.html(_audio_html, height=80)

# Browser camera controller. It only captures when device-camera mode is active.
components.html(
    f"""
    <div style="font-family:sans-serif;color:#ddd;background:#161616;
                border-radius:10px;padding:12px;margin:0 0 8px 0;">
      <div style="font-size:14px;font-weight:600;margin-bottom:6px;">
        Camera Source
      </div>
      <div id="camera-mode-status" style="font-size:12px;color:#aaa;">
        Preparing camera controls…
      </div>
      <video id="local-camera-preview" playsinline autoplay muted
             style="display:none;width:100%;margin-top:10px;border-radius:8px;
                    background:#000;"></video>
      <canvas id="local-camera-canvas" width="{PROC_W}" height="{PROC_H}"
              style="display:none;"></canvas>
    </div>
    <script>
      (() => {{
        const MODE = {json.dumps(source_key)};
        const CAPTURE_ENABLED = {json.dumps(running and source_key == "browser_camera")};
        const statusEl = document.getElementById('camera-mode-status');
        const previewEl = document.getElementById('local-camera-preview');
        const canvasEl = document.getElementById('local-camera-canvas');
        const ctx = canvasEl.getContext('2d');
        const parentHost = window.parent && window.parent.location
          ? window.parent.location.hostname
          : '';
        const secureContext = window.parent && 'isSecureContext' in window.parent
          ? window.parent.isSecureContext
          : window.isSecureContext;
        const host = parentHost || window.location.hostname || 'localhost';
        const proto = window.parent && window.parent.location
          ? window.parent.location.protocol
          : window.location.protocol || 'http:';
        const uploadUrl = proto + '//' + host + ':{MJPEG_PORT}/upload_frame?mode=' + MODE;

        let stream = null;
        let timer = null;
        let sending = false;

        async function stopCamera() {{
          if (timer) {{
            clearInterval(timer);
            timer = null;
          }}
          if (stream) {{
            stream.getTracks().forEach(track => track.stop());
            stream = null;
          }}
          previewEl.srcObject = null;
          previewEl.style.display = 'none';
        }}

        async function uploadFrame() {{
          if (!stream || sending || previewEl.readyState < 2) return;
          sending = true;
          ctx.drawImage(previewEl, 0, 0, canvasEl.width, canvasEl.height);
          const blob = await new Promise(resolve =>
            canvasEl.toBlob(resolve, 'image/jpeg', 0.72)
          );
          if (!blob) {{
            sending = false;
            return;
          }}

          try {{
            await fetch(uploadUrl, {{
              method: 'POST',
              headers: {{ 'Content-Type': 'image/jpeg' }},
              body: blob,
              cache: 'no-store',
            }});
          }} catch (err) {{
            statusEl.textContent = 'Could not upload device camera frames to the laptop.';
          }} finally {{
            sending = false;
          }}
        }}

        async function startCamera() {{
          if (!CAPTURE_ENABLED) {{
            statusEl.textContent = MODE === 'browser_camera'
              ? 'Turn on Run to start the device camera.'
              : 'IP Webcam feed is selected.';
            await stopCamera();
            return;
          }}

          if (!secureContext) {{
            statusEl.textContent =
              'Device camera mode needs HTTPS or localhost in most mobile browsers.';
            await stopCamera();
            return;
          }}

          try {{
            stream = await navigator.mediaDevices.getUserMedia({{
              video: {{
                facingMode: {{ ideal: 'environment' }},
                width: {{ ideal: {PROC_W} }},
                height: {{ ideal: {PROC_H} }}
              }},
              audio: false
            }});
            previewEl.srcObject = stream;
            previewEl.style.display = 'block';
            statusEl.textContent = 'Using this device camera and uploading frames for processing.';
            timer = setInterval(uploadFrame, 180);
          }} catch (err) {{
            statusEl.textContent = 'Camera access was denied or unavailable on this device.';
            await stopCamera();
          }}
        }}

        startCamera();
        window.addEventListener('beforeunload', stopCamera);
      }})();
    </script>
    """,
    height=260 if source_key == "browser_camera" else 110,
)

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
    if source_key == "ip_webcam":
        video_source.use_ip_webcam(url)
    else:
        video_source.use_browser_camera()
    processor.start()

    radar_slot.markdown(render_radar(buffer.objects), unsafe_allow_html=True)
    fps_slot.metric("FPS", f"{buffer.fps:.1f}")

    time.sleep(0.5)
    st.rerun()

else:
    processor.stop()
    video_source.stop()
    write_scene_json([])  # silence audio
    radar_slot.markdown(render_radar([]), unsafe_allow_html=True)
    fps_slot.metric("FPS", "—")
