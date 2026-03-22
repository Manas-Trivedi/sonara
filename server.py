"""
Sonara — FastAPI backend.

The laptop is a headless server. The phone browser is the entire product.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8501 \
        --ssl-keyfile ./.certs/key.pem --ssl-certfile ./.certs/cert.pem

Then open https://<laptop-ip>:8501 on the phone.
(Accept the self-signed cert warning once: Advanced → Proceed.)
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import scene
from depth import DepthEstimator
from detection import Detector

# ---------------------------------------------------------------------------
PROC_W, PROC_H = 384, 288
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
SCENE_JSON = STATIC_DIR / "scene.json"
CERTS_DIR = BASE_DIR / ".certs"
CERT_PATH = CERTS_DIR / "cert.pem"
KEY_PATH = CERTS_DIR / "key.pem"
STATIC_DIR.mkdir(exist_ok=True)
CERTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Video source — IP Webcam pull OR browser-camera push
# ---------------------------------------------------------------------------
class VideoSource:
    """Provides the latest raw frame from either an MJPEG URL or phone POSTs."""

    def __init__(self):
        self._mode = "browser_camera"
        self._url = None
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def mode(self):
        return self._mode

    def use_ip_webcam(self, url: str):
        restart = self._mode != "ip_webcam" or self._url != url
        self._mode = "ip_webcam"
        if not restart and self._thread and self._thread.is_alive():
            return
        self._stop_reader()
        self._clear()
        self._url = url
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def use_browser_camera(self):
        self._mode = "browser_camera"
        self._stop_reader()
        self._clear()

    def stop(self):
        self._stop_reader()
        self._clear()

    def push_frame(self, frame: np.ndarray):
        """Accept a decoded frame from the phone's POST /frame."""
        with self._lock:
            self._frame = frame

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _clear(self):
        with self._lock:
            self._frame = None

    def _stop_reader(self):
        self._stop.set()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None

    @staticmethod
    def _candidate_urls(url: str) -> list[str]:
        """Try common IP Webcam MJPEG endpoints when users provide only host:port."""
        parsed = urlparse(url)
        candidates = [url]

        if parsed.scheme and parsed.netloc:
            base = f"{parsed.scheme}://{parsed.netloc}"
            path = parsed.path or ""
            query = f"?{parsed.query}" if parsed.query else ""

            if path in ("", "/"):
                candidates.extend([
                    f"{base}/video{query}",
                    f"{base}/videofeed{query}",
                    f"{base}/mjpegfeed{query}",
                    f"{base}/mjpeg{query}",
                ])
            elif path.rstrip("/") != "/video":
                candidates.append(f"{base}/video{query}")

        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        return ordered

    @staticmethod
    def _open_capture(url: str):
        for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
            try:
                cap = cv2.VideoCapture(url, backend)
            except TypeError:
                cap = cv2.VideoCapture(url)

            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap

            if cap is not None:
                cap.release()
        return None

    def _reader_loop(self):
        urls = self._candidate_urls(self._url or "")
        if not urls:
            return

        url_i = 0
        failed_reads = 0

        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._open_capture(urls[url_i])
                if self._cap is None:
                    url_i = (url_i + 1) % len(urls)
                    time.sleep(0.25)
                    continue

            ok, f = self._cap.read()
            if not ok or f is None or f.size == 0:
                failed_reads += 1
                if failed_reads >= 12:
                    self._cap.release()
                    self._cap = None
                    failed_reads = 0
                    url_i = (url_i + 1) % len(urls)
                    time.sleep(0.1)
                else:
                    time.sleep(0.03)
                continue

            failed_reads = 0
            with self._lock:
                self._frame = f


# ---------------------------------------------------------------------------
# Shared annotated-frame buffer
# ---------------------------------------------------------------------------
class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self.objects: list[dict] = []
        self.fps: float = 0.0
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

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg if self._jpeg is not None else self._placeholder


# ---------------------------------------------------------------------------
# Processing thread
# ---------------------------------------------------------------------------
class Processor:
    DEPTH_EVERY = 3
    YOLO_EVERY = 2

    def __init__(self, video_source, depth_model, detector, buffer):
        self.video_source = video_source
        self.depth_model = depth_model
        self.detector = detector
        self.buffer = buffer
        self.running = threading.Event()
        self.quantum = False
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
        last_dets: list[dict] = []
        t_prev = time.time()

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

            write_scene_json(objects, self.quantum)
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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_annotated(frame_bgr, depth_map, objects):
    out = frame_bgr.copy()
    if depth_map is not None:
        heat = (depth_map * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)
        out = cv2.addWeighted(out, 0.6, heat, 0.4, 0)
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label_name = (o["label"] if o.get("label") not in (None, "", "None", "none")
                      else f"{o['zone']}-{o['level']}")
        label = f"{label_name} {o['distance_m']:.1f}m"
        cv2.putText(out, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if o["danger"]:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return out


def write_scene_json(objects, quantum=False):
    tmp = SCENE_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"objects": objects, "ts": time.time(), "quantum": quantum}, f)
    os.replace(tmp, SCENE_JSON)


# ---------------------------------------------------------------------------
# Self-signed cert (getUserMedia requires HTTPS on non-localhost)
# ---------------------------------------------------------------------------
def ensure_self_signed_cert():
    cert = CERT_PATH
    key = KEY_PATH
    if not cert.exists() or not key.exists():
        print("Generating self-signed certificate…", flush=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048",
             "-keyout", str(key), "-out", str(cert), "-days", "365",
             "-nodes", "-subj", "/CN=sonara"],
            check=True,
        )


ensure_self_signed_cert()


# ---------------------------------------------------------------------------
# Get local IP address for easy mobile access
# ---------------------------------------------------------------------------
def get_local_ip():
    """Get the laptop's local IP address."""
    try:
        # Connect to a remote host to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
video_source = VideoSource()
buffer = FrameBuffer()
processor: Processor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    print("Loading MiDaS depth model…", flush=True)
    depth_model = DepthEstimator()
    print("Loading YOLOv8n…", flush=True)
    detector = Detector()
    processor = Processor(video_source, depth_model, detector, buffer)
    write_scene_json([])
    ip = get_local_ip()
    print(f"Ready. Open this URL on your phone:\n  ➜ https://{ip}:8501", flush=True)
    yield
    processor.stop()
    video_source.stop()


app = FastAPI(title="Sonara", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/stream")
async def stream():
    async def gen():
        while True:
            jpeg = buffer.latest_jpeg()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(0.033)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/frame")
async def receive_frame(request: Request):
    body = await request.body()
    arr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is not None:
        video_source.push_frame(frame)
    return {"ok": frame is not None}


@app.post("/control")
async def control(payload: dict):
    action = payload.get("action")
    source = payload.get("source")
    url = payload.get("url")
    quantum = payload.get("quantum")

    if quantum is not None and processor:
        processor.quantum = bool(quantum)

    if action == "start":
        if source == "ip_webcam" and url:
            video_source.use_ip_webcam(url)
        else:
            video_source.use_browser_camera()
        if processor:
            processor.start()
    elif action == "stop":
        if processor:
            processor.stop()
        video_source.stop()
        write_scene_json([])

    return {"ok": True}


@app.get("/status")
async def status():
    return {
        "running": processor.running.is_set() if processor else False,
        "fps": round(buffer.fps, 1),
        "source": video_source.mode,
        "quantum": processor.quantum if processor else False,
    }
