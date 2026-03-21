# Sonara

Real-time spatial-audio navigation aid. Points a phone camera at the world,
runs depth estimation + object detection on a laptop, and plays back
**stereo tones whose pitch = distance and pan = direction**, plus **haptic
buzzes** when something is dangerously close.

Built for a 24-hour hackathon. Prioritises *working* over *pretty*.

---

## How it works

```
Phone camera ──WiFi/Browser──▶ Laptop (MiDaS + YOLOv8n) ──WiFi──▶ Phone browser
 (IP Webcam or website camera)     depth + detection         (audio + vibration)
```

- **MiDaS** gives per-pixel depth from a single RGB frame
- **YOLOv8n** finds objects
- `scene.py` fuses them → top-3 nearest objects with `{label, proximity, x_pos}`
- Browser JS turns each object into a sine tone:
  - closer = higher pitch (100 Hz far → 1200 Hz touching)
  - left/right = stereo pan
  - anything under ~1 m → 1800 Hz click + phone vibrates

---

## Setup

### 1. Choose a camera source

You now have two ways to feed video into Sonara:

**Option A: IP Webcam / MJPEG app**

- **Android:** install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam).
  Open it → scroll down → **Start server**. Note the URL it shows
  (e.g. `http://192.168.1.42:8080`). The video stream is at `…/video`.
- **iPhone:** use any app that exposes an MJPEG URL. Same idea.

**Option B: Use the website directly on your phone**

- Open Sonara in the phone browser.
- Switch **Video Source** to **This device camera**.
- Tap **Run** and allow camera access when the browser asks.
- Sonara will upload frames from the browser camera to the laptop for processing.
- Note: many mobile browsers only allow this on **HTTPS** (or `localhost`), so plain LAN `http://...` may still require IP Webcam.

### 2. Same WiFi

Phone and laptop **must** be on the same WiFi network. Corporate/campus
networks with client isolation will break this — use a hotspot if needed.

### 3. Install deps (laptop)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads MiDaS (~100 MB) and YOLOv8n (~6 MB). Give it a minute.

### 4. Run

```bash
streamlit run app.py --server.address=0.0.0.0
```

Streamlit prints a **Network URL** like `http://192.168.1.10:8501`.

If you want to use **This device camera** in the browser, plain LAN `http://`
is often not enough on mobile. In that case run Streamlit over **HTTPS**:

```bash
streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.sslCertFile=/path/to/cert.pem \
  --server.sslKeyFile=/path/to/key.pem
```

### 5. Open on the **phone**, not the laptop

This is the important bit. The **Vibration API only vibrates the device
running the browser**. So:

1. On the **phone**, open Chrome/Firefox and go to the Network URL from step 4.
2. Tap the orange **Enable Audio + Haptics** button.
3. Pick a source:
   - **IP Webcam feed**: paste the MJPEG URL (ending in `/video`) and turn on **Start Camera Processing** in the sidebar.
   - **This device camera**: switch modes, turn on **Start Camera Processing** in the sidebar, then allow camera access.

Keep the laptop browser open too if you want judges to see the annotated
video feed on a big screen — audio/haptics just won't fire there.

---

## Files

| File | What |
|------|------|
| `app.py` | Streamlit entry point, main loop, rendering |
| `depth.py` | MiDaS wrapper (auto-picks DPT_Large on GPU, MiDaS_small on CPU) |
| `detection.py` | YOLOv8n wrapper |
| `scene.py` | Fuses depth + detections → top-3 objects. **Run this standalone first** (`python scene.py`) to verify the data contract with mock data. |
| `audio_component.html` | Web Audio + Vibration JS, polls `static/scene.json` every 100 ms |
| `.streamlit/config.toml` | Enables static file serving + binds to `0.0.0.0` |

---

## Troubleshooting

- **No audio?** Browsers block `AudioContext` until a user gesture. Tap the
  orange button. On iOS Safari you may also need to un-mute the ringer.
- **No vibration?** Only works on Android Chrome/Firefox. iOS Safari does
  not implement `navigator.vibrate`. Also won't fire if the phone is on silent.
- **Low FPS?** You're CPU-bound on MiDaS. It already skips every other frame;
  if still slow, edit `depth.py` and pass `force_small=True`.
- **Can't reach the stream?** `curl http://<phone-ip>:8080/video` from the
  laptop. If that fails, it's a network problem, not a code problem.
- **Device camera mode shows permission errors?** The phone browser must allow
  camera access, the page still needs to reach the laptop on the local network,
  and some browsers require **HTTPS** before `getUserMedia()` will work.
