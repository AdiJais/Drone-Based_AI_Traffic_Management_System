"""
AeroSignal — server.py
======================
Run:  python server.py
Open: http://localhost:5000

Key behaviours
──────────────
• Vehicle COUNT only increments when a vehicle crosses the invisible
  midframe line (sv.LineZone from yolov8_ip.py) — not on first detection.
• The line is NEVER drawn on the MJPEG stream sent to the browser.
• NMS + ByteTrack deduplication prevent the same car being counted twice.
• All live data is pushed to the browser via Server-Sent Events (/events).

Requirements
────────────
  pip install flask ultralytics supervision numpy opencv-python requests
"""

import threading
import time
import json
import queue
import os

import cv2
import numpy as np

from flask import Flask, Response, request, jsonify, send_from_directory
from collections import deque

# ═══════════════════════════════════════════════════════════════════════
# SUPERVISION / YOLO — version-safe import
# ═══════════════════════════════════════════════════════════════════════
try:
    from ultralytics import YOLO
    import supervision as sv

    # ByteTrack class name changed across supervision versions
    if hasattr(sv, "ByteTrack"):
        _TrackerCls = sv.ByteTrack          # supervision >= 0.18
    elif hasattr(sv, "ByteTracker"):
        _TrackerCls = sv.ByteTracker        # supervision < 0.18
    else:
        raise ImportError("supervision has neither ByteTrack nor ByteTracker")

    # Update method name also changed across versions
    _probe = _TrackerCls()
    if hasattr(_probe, "update_with_detections"):
        def _track(tracker, dets):
            return tracker.update_with_detections(dets)
    elif hasattr(_probe, "update"):
        def _track(tracker, dets):
            return tracker.update(dets)
    else:
        raise ImportError("Tracker has no recognised update method")
    del _probe

    # LineZone trigger method name also changed
    _lz_probe = sv.LineZone(start=sv.Point(0, 1), end=sv.Point(1, 1))
    if hasattr(_lz_probe, "trigger"):
        def _lz_trigger(lz, dets):
            lz.trigger(dets)
    elif hasattr(_lz_probe, "count"):
        def _lz_trigger(lz, dets):
            lz.count(dets)
    else:
        def _lz_trigger(lz, dets):
            pass    # graceful no-op
    del _lz_probe

    YOLO_AVAILABLE = True
    print(f"[OK] supervision {sv.__version__} — tracker: {_TrackerCls.__name__}")

except ImportError as _err:
    YOLO_AVAILABLE = False
    sv             = None
    _TrackerCls    = None
    def _track(tracker, dets): return dets
    def _lz_trigger(lz, dets): pass
    print(f"[WARN] YOLO/supervision unavailable: {_err}")
    print("       pip install ultralytics supervision")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
CFG = {
    "VIDEO_PATH":              "http://10.28.139.225:8080/video",
    "VEHICLE_MODEL_PATH":      "yolov8x.pt",
    "POTHOLE_MODEL_PATH":      "pothole.pt",
    "VEHICLE_CLASSES":         [2, 3, 5, 7],   # car=2, motorcycle=3, bus=5, truck=7
    "CONFIDENCE":              0.4,
    "LOW_TRAFFIC_THRESHOLD":   5,
    "LOW_TRAFFIC_SECONDS":     5.0,
    "MIN_GREEN_SECONDS":       10.0,
    "MAX_GREEN_SECONDS":       60.0,
    "YELLOW_SECONDS":          3.0,
    "JAM_STOPPED_RATIO":       0.75,
    "JAM_SECONDS":             5.0,
    "STOPPED_DISPLACEMENT_PX": 8,
    "STOPPED_HISTORY_FRAMES":  10,
    "ROLLING_WINDOW_FRAMES":   15,
    "LANE_NAMES":              ["Lane 1", "Lane 2", "Lane 3", "Lane 4"],
    "NMS_IOU_THRESHOLD":       0.45,
    "AGNOSTIC_NMS":            True,
    "MIN_BOX_AREA":            800,
    # Fraction of frame height where the invisible counting line sits.
    # 0.5 = midframe, matching yolov8_ip.py (LINE_START/LINE_END at FRAME_HEIGHT//2)
    "LINE_POSITION":           0.5,
}

# ═══════════════════════════════════════════════════════════════════════
# FLASK + GLOBALS
# ═══════════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=".")

latest_frame   = None
frame_lock     = threading.Lock()
sse_clients    = []
sse_lock       = threading.Lock()
stream_running = False
stream_thread  = None
vehicle_model  = None
pothole_model  = None

# Invisible line counter and tracker (reinitialised per stream session)
line_zone         = None
byte_tracker      = None
session_in_count  = 0
session_out_count = 0

# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════
def load_models():
    global vehicle_model, pothole_model
    if not YOLO_AVAILABLE:
        return
    try:
        print(f"[MODEL] Loading {CFG['VEHICLE_MODEL_PATH']} ...")
        vehicle_model = YOLO(CFG["VEHICLE_MODEL_PATH"])
        print("[MODEL] Vehicle model ready")
    except Exception as e:
        print(f"[MODEL] Vehicle model failed: {e}")
    try:
        print(f"[MODEL] Loading {CFG['POTHOLE_MODEL_PATH']} ...")
        pothole_model = YOLO(CFG["POTHOLE_MODEL_PATH"])
        print("[MODEL] Pothole model ready")
    except Exception as e:
        print(f"[MODEL] Pothole model failed: {e}")

# ═══════════════════════════════════════════════════════════════════════
# ROLLING COUNTER  (mirrors yolov8_ip.py RollingCounter)
# ═══════════════════════════════════════════════════════════════════════
class RollingCounter:
    def __init__(self, window=15):
        self._h = deque(maxlen=window)
    def update(self, v):
        self._h.append(v)
        return sum(self._h) / len(self._h)
    def reset(self):
        self._h.clear()

rolling = RollingCounter(CFG["ROLLING_WINDOW_FRAMES"])

# ═══════════════════════════════════════════════════════════════════════
# MOTION TRACKER  (mirrors yolov8_ip.py MotionTracker)
# ═══════════════════════════════════════════════════════════════════════
class MotionTracker:
    def __init__(self):
        self.history = {}

    def update(self, tracker_ids, xyxy):
        result = {}
        for tid, box in zip(tracker_ids, xyxy):
            tid = int(tid)
            cx  = float((box[0] + box[2]) / 2)
            cy  = float((box[1] + box[3]) / 2)
            if tid not in self.history:
                self.history[tid] = deque(maxlen=CFG["STOPPED_HISTORY_FRAMES"])
            self.history[tid].append((cx, cy))
            h = self.history[tid]
            if len(h) >= 2:
                dx   = h[-1][0] - h[0][0]
                dy   = h[-1][1] - h[0][1]
                disp = (dx*dx + dy*dy) ** 0.5
                result[tid] = {
                    "stopped":      disp < CFG["STOPPED_DISPLACEMENT_PX"],
                    "displacement": round(disp, 1),
                }
            else:
                result[tid] = {"stopped": False, "displacement": 0.0}
        active = {int(t) for t in tracker_ids}
        self.history = {k: v for k, v in self.history.items() if k in active}
        return result

motion_tracker = MotionTracker()

# ═══════════════════════════════════════════════════════════════════════
# SSE BROADCAST
# ═══════════════════════════════════════════════════════════════════════
def broadcast(event_type, data):
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)

# ═══════════════════════════════════════════════════════════════════════
# SESSION INIT — fresh LineZone + tracker every time a stream connects
# ═══════════════════════════════════════════════════════════════════════
def _init_session(frame_w, frame_h):
    """
    Create invisible sv.LineZone at LINE_POSITION * frame_h.
    This exactly mirrors yolov8_ip.py:
        LINE_START = sv.Point(0, FRAME_HEIGHT // 2)
        LINE_END   = sv.Point(FRAME_WIDTH, FRAME_HEIGHT // 2)
    The line is used for counting only — never drawn on output frames.
    """
    global byte_tracker, line_zone
    global session_in_count, session_out_count
    global motion_tracker, rolling

    motion_tracker    = MotionTracker()
    rolling           = RollingCounter(CFG["ROLLING_WINDOW_FRAMES"])
    session_in_count  = 0
    session_out_count = 0

    byte_tracker = _TrackerCls() if _TrackerCls is not None else None

    if sv is not None:
        y = int(frame_h * CFG["LINE_POSITION"])
        line_zone = sv.LineZone(
            start = sv.Point(0,       y),
            end   = sv.Point(frame_w, y),
        )
        print(f"[LINE] Invisible counting line: y={y}px "
              f"({int(CFG['LINE_POSITION']*100)}% of {frame_h}px frame height)")
    else:
        line_zone = None

# ═══════════════════════════════════════════════════════════════════════
# NMS HELPERS
# ═══════════════════════════════════════════════════════════════════════
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_ICONS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    if not inter:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def _fallback_nms(dets, iou_thr, agnostic):
    if not len(dets):
        return dets
    boxes = dets.xyxy
    confs = dets.confidence
    cids  = dets.class_id
    order = confs.argsort()[::-1]
    keep  = []
    supp  = set()
    for i, idx in enumerate(order):
        if idx in supp:
            continue
        keep.append(idx)
        for jdx in order[i+1:]:
            if jdx not in supp and (agnostic or cids[idx]==cids[jdx]) \
               and _iou(boxes[idx], boxes[jdx]) > iou_thr:
                supp.add(jdx)
    return dets[np.array(keep)]

# ═══════════════════════════════════════════════════════════════════════
# CORE DETECTION
# ═══════════════════════════════════════════════════════════════════════
frame_counter = 0

def run_detection_on_frame(frame):
    """
    Runs YOLO + NMS + ByteTrack + invisible LineZone on one frame.

    Vehicle count (lineIn / lineOut) only increments when a tracked
    vehicle centroid crosses the invisible horizontal line — exactly
    matching the sv.LineZone logic in yolov8_ip.py.

    Returns a dict broadcast to the browser via SSE /events.
    """
    global frame_counter, session_in_count, session_out_count
    frame_counter += 1

    if not YOLO_AVAILABLE or vehicle_model is None or byte_tracker is None:
        return None

    try:
        # 1. YOLO inference — NMS baked in at inference level
        results = vehicle_model(
            frame,
            conf         = CFG["CONFIDENCE"],
            iou          = CFG["NMS_IOU_THRESHOLD"],
            agnostic_nms = CFG["AGNOSTIC_NMS"],
            verbose      = False,
        )[0]

        dets = sv.Detections.from_ultralytics(results)

        # 2. Vehicle class filter
        dets = dets[np.isin(dets.class_id, CFG["VEHICLE_CLASSES"])]

        # 3. Minimum box area — remove background noise
        if CFG["MIN_BOX_AREA"] > 0 and len(dets):
            boxes = dets.xyxy
            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
            dets  = dets[areas >= CFG["MIN_BOX_AREA"]]

        # 4. Post-inference NMS — second pass catches any survivors
        if len(dets):
            if hasattr(dets, "with_nms"):
                dets = dets.with_nms(
                    threshold      = CFG["NMS_IOU_THRESHOLD"],
                    class_agnostic = CFG["AGNOSTIC_NMS"],
                )
            else:
                dets = _fallback_nms(dets, CFG["NMS_IOU_THRESHOLD"], CFG["AGNOSTIC_NMS"])

        # 5. ByteTrack — assigns persistent IDs, one per physical vehicle
        dets = _track(byte_tracker, dets)

        # 6. Tracker-ID dedup (safety net)
        if dets.tracker_id is not None and len(dets):
            seen, keep = {}, []
            for i, tid in enumerate(dets.tracker_id):
                tid = int(tid)
                if tid not in seen:
                    seen[tid] = i
                    keep.append(i)
                elif dets.confidence[i] > dets.confidence[seen[tid]]:
                    keep[keep.index(seen[tid])] = i
                    seen[tid] = i
            dets = dets[np.array(keep)]

        # 7. Invisible LineZone crossing counter
        #    ─────────────────────────────────────────────────────────────
        #    sv.LineZone internally records which tracker_ids have already
        #    crossed (using the persistent IDs from ByteTrack), so each
        #    vehicle is counted exactly ONCE no matter how many frames
        #    its bounding box overlaps the line position.
        #
        #    We snapshot in_count / out_count before and after to know
        #    how many NEW vehicles crossed THIS frame.
        #    ─────────────────────────────────────────────────────────────
        prev_in  = line_zone.in_count  if line_zone else 0
        prev_out = line_zone.out_count if line_zone else 0

        if line_zone is not None and len(dets):
            _lz_trigger(line_zone, dets)   # counting only — NO drawing

        new_in  = (line_zone.in_count  - prev_in)  if line_zone else 0
        new_out = (line_zone.out_count - prev_out) if line_zone else 0

        session_in_count  += new_in
        session_out_count += new_out

        # 8. Motion tracking (stopped vs moving)
        tracker_ids = dets.tracker_id if dets.tracker_id is not None else []
        stopped_map = {}
        if len(dets) and len(tracker_ids):
            stopped_map = motion_tracker.update(tracker_ids, dets.xyxy)

        raw_count     = len(dets)
        stopped_count = sum(
            1 for tid in tracker_ids
            if stopped_map.get(int(tid), {}).get("stopped", False)
        )
        smoothed = rolling.update(raw_count)

        # 9. Build SSE payload
        type_counts = {k: 0 for k in CLASS_NAMES.values()}
        tracks_out  = []
        for conf, cid, tid in zip(
            dets.confidence,
            dets.class_id,
            tracker_ids if len(tracker_ids) else [None] * raw_count,
        ):
            name = CLASS_NAMES.get(int(cid), "vehicle")
            type_counts[name] = type_counts.get(name, 0) + 1
            motion = stopped_map.get(int(tid) if tid is not None else -1, {})
            tracks_out.append({
                "id":           int(tid) if tid is not None else -1,
                "className":    name,
                "classIcon":    CLASS_ICONS.get(int(cid), "vehicle"),
                "conf":         round(float(conf), 2),
                "stopped":      motion.get("stopped", False),
                "displacement": motion.get("displacement", 0.0),
            })

        fh, fw = frame.shape[:2]
        boxes_out = []
        for box, cid, tid in zip(
            dets.xyxy,
            dets.class_id,
            tracker_ids if len(tracker_ids) else [None] * raw_count,
        ):
            motion = stopped_map.get(int(tid) if tid is not None else -1, {})
            boxes_out.append({
                "x":       round(float(box[0]) / fw, 4),
                "y":       round(float(box[1]) / fh, 4),
                "w":       round(float(box[2]-box[0]) / fw, 4),
                "h":       round(float(box[3]-box[1]) / fh, 4),
                "label":   f"#{tid} {CLASS_NAMES.get(int(cid), '?')}",
                "stopped": motion.get("stopped", False),
            })

        return {
            # Line-crossing counts — THE authoritative vehicle count
            "lineIn":       session_in_count,
            "lineOut":      session_out_count,
            "newIn":        new_in,      # new crossings this frame only
            "newOut":       new_out,
            # Frame-level counts — used by signal machine + jam meter
            "rawCount":     raw_count,
            "stoppedCount": stopped_count,
            "smoothed":     round(smoothed, 2),
            "vehicleTypes": type_counts,
            "tracks":       tracks_out,
            "boxes":        boxes_out,
        }

    except Exception as e:
        print(f"[DETECTION] Error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════
# STREAM WORKER
# ═══════════════════════════════════════════════════════════════════════
def stream_worker():
    global latest_frame, stream_running

    url = CFG["VIDEO_PATH"]
    print(f"[STREAM] Connecting to {url} ...")
    broadcast("status", {"connected": False, "msg": f"Connecting to {url}..."})

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[STREAM] Cannot open: {url}")
        broadcast("status", {"connected": False, "msg": f"Cannot connect to {url}"})
        stream_running = False
        return

    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[STREAM] {fw}x{fh} @ {fps_src:.1f} fps")

    # Initialise invisible line + trackers sized to this stream
    _init_session(fw, fh)

    broadcast("status", {"connected": True, "msg": f"Stream connected: {url}"})
    broadcast("streamInfo", {
        "width":   fw,
        "height":  fh,
        "fps":     fps_src,
        "lineY":   CFG["LINE_POSITION"],   # sent so frontend knows line position
    })
    print("[STREAM] Connected")

    fps_counter  = 0
    fps_ts       = time.time()
    det_interval = 3   # run YOLO every N frames; raise to 5 on slow hardware

    while stream_running:
        ok, frame = cap.read()
        if not ok:
            print("[STREAM] Frame read failed — reconnecting...")
            broadcast("status", {"connected": False, "msg": "Stream lost — reconnecting..."})
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                broadcast("status", {"connected": False, "msg": f"Reconnect failed: {url}"})
                break
            new_fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            new_fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _init_session(new_fw, new_fh)
            broadcast("status", {"connected": True, "msg": "Reconnected"})
            continue

        # Encode CLEAN frame — NO line drawn
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buf.tobytes()

        fps_counter += 1
        now = time.time()
        if now - fps_ts >= 1.0:
            broadcast("fps", {"fps": round(fps_counter / (now - fps_ts), 1)})
            fps_counter = 0
            fps_ts      = now

        if fps_counter % det_interval == 0:
            payload = run_detection_on_frame(frame)
            if payload:
                broadcast("detection", payload)

    cap.release()
    stream_running = False
    broadcast("status", {"connected": False, "msg": "Stream stopped"})
    print("[STREAM] Stopped")

# ═══════════════════════════════════════════════════════════════════════
# MJPEG GENERATOR
# ═══════════════════════════════════════════════════════════════════════
def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame + b"\r\n"
            )
        time.sleep(0.033)

# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    base = os.path.dirname(os.path.abspath(__file__))
    html = "aerosignal_integrated.html"
    if not os.path.exists(os.path.join(base, html)):
        return f"<h2>{html} not found next to server.py</h2>", 404
    return send_from_directory(base, html)

@app.route("/stream")
def stream_route():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/events")
def events():
    q = queue.Queue(maxsize=100)
    with sse_lock:
        sse_clients.append(q)

    def generator():
        try:
            while True:
                try:
                    yield q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(
        generator(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.route("/connect", methods=["POST"])
def connect():
    global stream_running, stream_thread
    data = request.json or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "No URL provided"}), 400

    stream_running = False
    if stream_thread and stream_thread.is_alive():
        stream_thread.join(timeout=3)

    CFG["VIDEO_PATH"] = url
    stream_running    = True
    stream_thread     = threading.Thread(target=stream_worker, daemon=True)
    stream_thread.start()
    return jsonify({"ok": True, "url": url})

@app.route("/disconnect", methods=["POST"])
def disconnect():
    global stream_running
    stream_running = False
    return jsonify({"ok": True})

@app.route("/config", methods=["GET", "POST"])
def config_route():
    if request.method == "GET":
        return jsonify(CFG)
    data = request.json or {}
    for key, val in data.items():
        if key in CFG:
            try:
                CFG[key] = type(CFG[key])(val)
            except Exception:
                CFG[key] = val
    rolling._h = deque(rolling._h, maxlen=int(CFG["ROLLING_WINDOW_FRAMES"]))
    broadcast("config", CFG)
    return jsonify({"ok": True, "config": CFG})

@app.route("/status")
def status():
    return jsonify({
        "streamRunning":  stream_running,
        "streamUrl":      CFG["VIDEO_PATH"],
        "yoloAvailable":  YOLO_AVAILABLE,
        "vehicleModel":   CFG["VEHICLE_MODEL_PATH"],
        "potholeModel":   CFG["POTHOLE_MODEL_PATH"],
        "lineIn":         session_in_count,
        "lineOut":        session_out_count,
        "clients":        len(sse_clients),
    })

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  AeroSignal Server")
    print("=" * 60)
    print("  Dashboard  ->  http://localhost:5000")
    print("  Stream     ->  http://localhost:5000/stream")
    print("  Events     ->  http://localhost:5000/events")
    print("  Config     ->  http://localhost:5000/config")
    print("  Status     ->  http://localhost:5000/status")
    print("=" * 60)

    if not YOLO_AVAILABLE:
        print("\n  [!] YOLO not available - stream proxy will still work.")
        print("      Install: pip install ultralytics supervision\n")
    else:
        threading.Thread(target=load_models, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
