# Sonara

Sonara is a hackathon prototype for assistive navigation that turns a live camera feed into spatial audio and haptic cues. It uses monocular depth estimation plus object detection to help a user understand where nearby obstacles are, then plays directional sound on the phone and triggers vibration when something is too close.

Built by team `Zombie` for a hackathon sprint, Sonara is intentionally optimized for a live demo: fast setup, clear feedback, and a pipeline that still works even when object detection misses something.

## What It Does

- Estimates scene depth from a single RGB camera feed using MiDaS.
- Detects known objects with YOLOv8n for extra semantic context.
- Splits the view into a `3 x 2` grid: left, center, right x upper, lower.
- Computes proximity for all 6 zones, even if no object label is detected.
- Converts the nearest active zones into 3D-positioned audio cues in the browser.
- Triggers a danger click and phone vibration for very close obstacles.
- Supports both an external `IP Webcam` feed and the phone's own browser camera.
- Streams annotated output back for easier debugging and judging.

## Why The Current Version Is Better

The older prototype focused on only a few detected objects. The current build is more reliable for obstacle awareness because depth drives the experience first, and detection labels are layered on top when available.

That means:

- Sonara still reacts when YOLO sees nothing.
- The user gets consistent left/center/right and upper/lower guidance.
- Nearby walls, furniture, and unlabelled obstacles can still trigger warnings.
- The demo is easier to explain because every part of the frame maps to a zone.

## How It Works

```text
Phone camera or IP Webcam
        |
        v
Laptop processing app (Streamlit)
  - MiDaS depth estimation
  - YOLOv8n object detection
  - 6-zone scene analysis
        |
        +--> annotated MJPEG video stream
        |
        +--> static/scene.json with live obstacle state
                     |
                     v
Phone browser audio component
  - 3D spatial audio
  - danger beep
  - vibration feedback
```

## Core Pipeline

1. A frame comes from either an MJPEG stream or the phone browser camera.
2. The frame is resized and passed through MiDaS to estimate relative depth.
3. YOLOv8n detects objects and provides labels where possible.
4. `scene.py` divides the depth map into 6 regions and scores each region by proximity.
5. Each region becomes a structured output with:
   `zone`, `level`, `proximity`, `distance_m`, `label`, `bbox`, `danger`
6. The app writes this state to [`static/scene.json`](/Users/manastrivedi/code/sonara/static/scene.json).
7. The browser polls that file and sonifies the nearest active zones.

## Key Features

- `Depth-first obstacle awareness`
  Sonara does not depend entirely on object classes. If the model sees "something near" but cannot name it, the system still warns the user.

- `6-zone spatial mapping`
  The scene is divided into upper/lower and left/center/right regions, which makes the audio cues more intuitive during movement.

- `Live spatial audio`
  The browser uses Web Audio panners to position sounds in space. Nearer objects sound more urgent, and left/right placement matches the obstacle direction.

- `Danger haptics`
  High-risk zones trigger a sharper alert tone and vibration on supported phones.

- `Two camera modes`
  Use an IP camera app for easier local-network streaming, or use the phone's own camera directly from the browser.

- `Judge-friendly visualization`
  The laptop can display the annotated stream with heatmap overlays, zone boxes, labels, and estimated distances while the phone handles audio and haptics.

## Tech Stack

- `Streamlit` for the UI and orchestration
- `OpenCV` for frame handling and rendering
- `PyTorch` + `MiDaS` for monocular depth estimation
- `Ultralytics YOLOv8n` for object detection
- `Web Audio API` for stereo/HRTF audio cues
- `Vibration API` for close-range warnings

## Project Structure

| File | Purpose |
|------|---------|
| [`app.py`](/Users/manastrivedi/code/sonara/app.py) | Main Streamlit app, camera control, processing loop, MJPEG server |
| [`depth.py`](/Users/manastrivedi/code/sonara/depth.py) | MiDaS model loading and normalized depth estimation |
| [`detection.py`](/Users/manastrivedi/code/sonara/detection.py) | YOLOv8n wrapper and detection formatting |
| [`scene.py`](/Users/manastrivedi/code/sonara/scene.py) | 6-zone scene analysis and danger scoring |
| [`audio_component.html`](/Users/manastrivedi/code/sonara/audio_component.html) | Browser-side audio and vibration logic |
| [`static/scene.json`](/Users/manastrivedi/code/sonara/static/scene.json) | Live obstacle state consumed by the frontend |
| [`requirements.txt`](/Users/manastrivedi/code/sonara/requirements.txt) | Python dependencies |

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the app

```bash
streamlit run app.py --server.address=0.0.0.0
```

Streamlit will print a local URL and a network URL. Open the network URL on your phone if you want audio and vibration on the actual device.

### 3. Choose a video source

#### Option A: IP Webcam feed

- Install an app like IP Webcam on the phone.
- Start the server in the app.
- Paste the MJPEG stream URL into Sonara, usually ending in `/video`.
- Turn on `Start Camera Processing`.

#### Option B: This device camera

- Open Sonara on the phone browser.
- Switch the source to `This device camera`.
- Turn on `Start Camera Processing`.
- Allow camera access when prompted.

Note: many mobile browsers require `HTTPS` or `localhost` for direct camera access. If plain local `http://` does not work, either serve Streamlit over HTTPS or use the IP Webcam route.

## Demo Flow

For a live hackathon demo:

1. Run the Streamlit app on the laptop.
2. Open the app on the phone.
3. Tap `Enable Audio + Haptics`.
4. Start camera processing.
5. Move the phone toward obstacles and let the judges hear the spatial cue changes.
6. Keep the laptop view visible so judges can see the annotated zones and distances in real time.

## Current Output Format

Each zone in the live scene state looks like this:

```json
{
  "zone": "left",
  "level": "upper",
  "proximity": 0.84,
  "distance_m": 0.78,
  "label": null,
  "bbox": [0, 0, 128, 144],
  "danger": true
}
```

This is the contract shared between the Python inference pipeline and the browser audio system.

## Limitations

- Depth is monocular and approximate, not true metric depth.
- Direct phone-camera mode may fail on non-HTTPS mobile sessions.
- Vibration support is browser and device dependent.
- Performance on CPU-only machines is lower than on systems with CUDA.
- This is a hackathon build, so robustness and accessibility testing are still limited.

## Team

`Zombie`

- Manas Trivedi
- Rahul Singh Jadoun
- Acid Singh

## Hackathon Note

Sonara was built as a rapid prototype to explore accessible, low-cost navigation assistance using devices people already have: a smartphone and a laptop. The focus was on making the feedback immediate and interpretable enough for a live demo, while keeping the stack simple enough to build fast during a hackathon.
