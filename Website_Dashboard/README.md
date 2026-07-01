# AeroSignal — Setup Guide

## Files in this folder
```
server.py                    ← Flask backend (run this)
aerosignal_integrated.html   ← Dashboard (served by server.py)
requirements.txt             ← Python dependencies
```

---

## Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

If you have a CUDA GPU (for faster YOLO):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2 — Place your YOLO model files here

Put these in the **same folder** as server.py:
- `yolov8x.pt`   — vehicle detection model
- `pothole.pt`   — pothole detection model

They will be downloaded automatically by ultralytics the first time if missing.

---

## Step 3 — Connect your IP Webcam

On your phone, install **IP Webcam** (Android) or **EpocCam** (iOS).  
Start the server on the phone, note the URL shown (e.g. `http://192.168.29.211:8080/video`).  
Your phone and laptop **must be on the same WiFi network**.

---

## Step 4 — Run the server

```bash
python server.py
```

You'll see:
```
  Dashboard  →  http://localhost:5000
  Stream     →  http://localhost:5000/stream
  Events     →  http://localhost:5000/events
```

---

## Step 5 — Open the dashboard

Go to **http://localhost:5000** in your browser.

Click **⚙ Connect**, enter your IP Webcam URL, and press Connect.

The server will:
1. Open the camera with OpenCV
2. Proxy the MJPEG stream to the browser at `/stream`
3. Run YOLOv8x + ByteTrack detection on each frame
4. Push vehicle counts, stopped ratios, and bounding boxes to the browser in real-time via Server-Sent Events (`/events`)

---

## How it works (architecture)

```
Phone Camera (IP Webcam)
        │
        │  MJPEG over HTTP (same WiFi)
        ▼
  server.py  (localhost:5000)
        │
        ├─ /stream  →  MJPEG proxy → browser <img>
        │               (no CORS issues)
        │
        ├─ /events  →  SSE stream → browser JS
        │               vehicle counts, boxes, signal state
        │
        ├─ /connect →  POST to start/change camera URL
        │
        └─ /config  →  GET/POST to read/write detection params
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot connect to camera URL" | Make sure phone and laptop are on the same WiFi. Try opening the camera URL in your browser first. |
| YOLO models not loading | Check that `.pt` files are in the same folder as server.py |
| Stream works but no detections | YOLO might still be loading (takes ~10s first time). Check terminal output. |
| Very slow detection | Lower confidence or change `det_interval` in server.py (line ~190) to skip more frames |
| Port 5000 in use | Change `port=5000` at the bottom of server.py |
