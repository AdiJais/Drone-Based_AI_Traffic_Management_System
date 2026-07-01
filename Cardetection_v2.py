from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import torch
import time
from collections import deque
from enum import Enum, auto

# =============================================================================
#  CONFIG  — edit everything in this block to match your setup
# =============================================================================

VIDEO_PATH   = 'http://10.28.139.225:8080/video'
MODEL_PATH   = 'yolov8x.pt'
WINDOW_TITLE = 'Drone Traffic Management'

# YOLO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# --- Lane names (drone will rotate through these one at a time) ---
LANE_NAMES = ['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4']

# --- Signal timing ---
LOW_TRAFFIC_THRESHOLD  = 5      # vehicles — below this triggers early-switch timer
LOW_TRAFFIC_SECONDS    = 5.0    # seconds below threshold before switching
MIN_GREEN_SECONDS      = 10.0   # minimum green time (prevent rapid flicker)
MAX_GREEN_SECONDS      = 60.0   # maximum green time (prevent lane starvation)
YELLOW_SECONDS         = 3.0    # how long yellow phase lasts

# --- Traffic jam detection ---
# Triggers an early switch when most vehicles are stopped (gridlock).
# Works independently of LOW_TRAFFIC_THRESHOLD so it fires even with many cars.
JAM_STOPPED_RATIO      = 0.75   # fraction of vehicles stopped to call it a jam
JAM_SECONDS            = 5.0    # seconds the jam must hold before switching

# --- Stopped-vehicle detection ---
STOPPED_DISPLACEMENT_PX = 8     # pixels of movement below which = stopped
STOPPED_HISTORY_FRAMES  = 10    # frames to track position history per ID

# --- Rolling average smoothing ---
ROLLING_WINDOW_FRAMES  = 15     # ~0.5s at 30fps

# --- Annotation ---
TRACE_LENGTH  = 60
BOX_THICKNESS = 2

# Signal trigger function — replace print() with serial/MQTT/HTTP as needed
# This is also where you send the "rotate drone" command to the next lane.
def trigger_signal_change(from_lane: str, to_lane: str):
    print(f"[SIGNAL] Switching GREEN: {from_lane} → {to_lane}")
    print(f"[DRONE]  Rotate to face: {to_lane}")
    # Example MQTT:
    #   client.publish("traffic/signal", json.dumps({"green": to_lane}))
    # Example serial to drone controller:
    #   ser.write(f"ROTATE:{to_lane}\n".encode())

# =============================================================================
#  MODEL SETUP
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

model = YOLO(MODEL_PATH).to(DEVICE)
model.fuse()
CLASS_NAMES = model.model.names

# =============================================================================
#  DYNAMIC VIDEO RESOLUTION
# =============================================================================

def get_video_properties(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fps    = fps if fps and fps > 0 else 25.0
    cap.release()
    return width, height, fps

FRAME_WIDTH, FRAME_HEIGHT, FPS = get_video_properties(VIDEO_PATH)
print(f"Source: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS:.1f} fps")

# =============================================================================
#  SINGLE FULL-FRAME ROI ZONE
#  The drone camera sees exactly one lane at a time — use the whole frame.
# =============================================================================

FULL_FRAME_POLYGON = np.array([
    [0,            0           ],
    [FRAME_WIDTH,  0           ],
    [FRAME_WIDTH,  FRAME_HEIGHT],
    [0,            FRAME_HEIGHT],
])

active_zone = sv.PolygonZone(polygon=FULL_FRAME_POLYGON)

# =============================================================================
#  FEATURE 1 — ROLLING AVERAGE SMOOTHER
# =============================================================================

class RollingCounter:
    """Smooths noisy per-frame vehicle counts over a sliding window."""
    def __init__(self, window: int = ROLLING_WINDOW_FRAMES):
        self._history = deque(maxlen=window)

    def update(self, count: int) -> float:
        self._history.append(count)
        return sum(self._history) / len(self._history)

    def reset(self):
        self._history.clear()

lane_smoother = RollingCounter()

# =============================================================================
#  FEATURE 3 — SIGNAL CHANGE LOGIC
# =============================================================================

class SignalChecker:
    """
    Two independent early-switch triggers:

    1. LOW TRAFFIC  — smoothed vehicle count below LOW_TRAFFIC_THRESHOLD
                      for LOW_TRAFFIC_SECONDS (lane is mostly empty).

    2. JAM          — fraction of stopped vehicles >= JAM_STOPPED_RATIO
                      for JAM_SECONDS (lane is gridlocked).

    Returns (should_switch, reason_string, countdown_or_None).
    """
    def __init__(self):
        self._low_since: float | None = None
        self._jam_since: float | None = None

    def check(self, smoothed_count: float, stopped_count: int,
              now: float) -> tuple[bool, str, float | None]:

        # ── Trigger 1: low traffic ──────────────────────────────────────────
        if smoothed_count < LOW_TRAFFIC_THRESHOLD:
            if self._low_since is None:
                self._low_since = now
            if now - self._low_since >= LOW_TRAFFIC_SECONDS:
                return True, "LOW TRAFFIC", 0.0
        else:
            self._low_since = None

        # ── Trigger 2: jam (most vehicles stopped) ──────────────────────────
        total = max(int(round(smoothed_count)), 1)
        stopped_ratio = stopped_count / total
        if stopped_ratio >= JAM_STOPPED_RATIO and smoothed_count >= 2:
            if self._jam_since is None:
                self._jam_since = now
            if now - self._jam_since >= JAM_SECONDS:
                return True, "TRAFFIC JAM", 0.0
        else:
            self._jam_since = None

        # ── Countdown for whichever timer is running ────────────────────────
        countdown = None
        if self._low_since is not None:
            countdown = max(0.0, LOW_TRAFFIC_SECONDS - (now - self._low_since))
        if self._jam_since is not None:
            jam_cd = max(0.0, JAM_SECONDS - (now - self._jam_since))
            countdown = jam_cd if countdown is None else min(countdown, jam_cd)

        return False, "", countdown

    def reset(self):
        self._low_since = None
        self._jam_since = None

    @property
    def active_countdown(self) -> float | None:
        now = time.time()
        cd = None
        if self._low_since is not None:
            cd = max(0.0, LOW_TRAFFIC_SECONDS - (now - self._low_since))
        if self._jam_since is not None:
            jam_cd = max(0.0, JAM_SECONDS - (now - self._jam_since))
            cd = jam_cd if cd is None else min(cd, jam_cd)
        return cd

# =============================================================================
#  FEATURE 4 — TRAFFIC LIGHT STATE MACHINE
# =============================================================================

class SignalState(Enum):
    GREEN  = auto()
    YELLOW = auto()

class TrafficStateMachine:
    """
    Cycles through lanes using a state machine with:
      - minimum green time  (prevents flickering)
      - maximum green time  (prevents lane starvation)
      - yellow phase        (realistic transition)
      - low-traffic trigger (early switch when lane is clear)

    Since the drone camera sees only one lane at a time, the state machine
    also manages a per-lane smoothed count stored from the previous visit.
    """
    def __init__(self, lane_names: list[str]):
        self._lanes         = lane_names
        self._idx           = 0
        self._state         = SignalState.GREEN
        self._phase_start   = time.time()
        self._checker       = SignalChecker()
        self._switch_reason = ""        # last reason a switch was triggered
        self._early_cd: float | None = None   # countdown from SignalChecker

        # Remember last known smoothed count for each lane (for HUD display)
        self._last_counts: dict[str, float] = {n: 0.0 for n in lane_names}

        print(f"[SIGNAL] Initial GREEN: {self.green_lane}")

    @property
    def green_lane(self) -> str:
        return self._lanes[self._idx]

    @property
    def state(self) -> SignalState:
        return self._state

    @property
    def switch_reason(self) -> str:
        return self._switch_reason

    @property
    def early_countdown(self) -> float | None:
        """Seconds until early-switch fires (None = no early trigger active)."""
        return self._early_cd

    def update(self, smoothed_count: float, stopped_count: int, now: float):
        """Call every frame with the smoothed count and stopped count for the current lane."""
        self._last_counts[self.green_lane] = smoothed_count
        elapsed = now - self._phase_start

        if self._state == SignalState.GREEN:
            force_switch = elapsed >= MAX_GREEN_SECONDS

            # Pass stopped_count into the dual-trigger checker
            early_switch, reason, cd = self._checker.check(
                smoothed_count, stopped_count, now
            )
            self._early_cd = cd if elapsed >= MIN_GREEN_SECONDS else None

            # Only act on early switch after minimum green time
            early_switch = early_switch and (elapsed >= MIN_GREEN_SECONDS)

            if force_switch or early_switch:
                self._switch_reason = "MAX TIME" if force_switch else reason
                print(f"[SIGNAL] {self.green_lane} → YELLOW ({self._switch_reason})")
                self._state       = SignalState.YELLOW
                self._phase_start = now
                self._early_cd    = None
                self._checker.reset()

        elif self._state == SignalState.YELLOW:
            self._early_cd = None
            if elapsed >= YELLOW_SECONDS:
                prev_lane = self.green_lane
                self._idx = (self._idx + 1) % len(self._lanes)
                self._state         = SignalState.GREEN
                self._phase_start   = now
                self._switch_reason = ""
                lane_smoother.reset()
                trigger_signal_change(prev_lane, self.green_lane)

    def countdown_to_switch(self, now: float) -> float:
        elapsed = now - self._phase_start
        if self._state == SignalState.YELLOW:
            return max(0.0, YELLOW_SECONDS - elapsed)
        time_to_max = max(0.0, MAX_GREEN_SECONDS - elapsed)
        if self._early_cd is not None:
            return min(time_to_max, self._early_cd)
        return time_to_max

    def last_counts(self) -> dict[str, float]:
        return dict(self._last_counts)

# =============================================================================
#  FEATURE 5 — STOPPED vs MOVING VEHICLE DETECTION
# =============================================================================

class MotionTracker:
    """
    Stores recent centre-point positions for each tracker ID.
    A vehicle is considered stopped if its displacement over
    STOPPED_HISTORY_FRAMES frames is below STOPPED_DISPLACEMENT_PX.
    """
    def __init__(self):
        self._history: dict[int, deque] = {}

    def update(self, tracker_ids: np.ndarray, boxes_xyxy: np.ndarray) -> dict[int, bool]:
        centres = ((boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2).astype(int)
        stopped = {}

        for tid, centre in zip(tracker_ids, centres):
            if tid not in self._history:
                self._history[tid] = deque(maxlen=STOPPED_HISTORY_FRAMES)
            self._history[tid].append(centre)

            history = self._history[tid]
            if len(history) >= 2:
                displacement = float(np.linalg.norm(
                    np.array(history[-1]) - np.array(history[0])
                ))
                stopped[tid] = displacement < STOPPED_DISPLACEMENT_PX
            else:
                stopped[tid] = False

        active = set(map(int, tracker_ids))
        self._history = {k: v for k, v in self._history.items() if k in active}
        return stopped

motion_tracker = MotionTracker()

# =============================================================================
#  TRACKING & ANNOTATION SETUP
# =============================================================================

LINE_START = sv.Point(0, FRAME_HEIGHT // 2)
LINE_END   = sv.Point(FRAME_WIDTH, FRAME_HEIGHT // 2)

byte_tracker        = sv.ByteTrack(frame_rate=int(FPS))
line_counter        = sv.LineZone(start=LINE_START, end=LINE_END)
line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
box_annotator       = sv.BoxAnnotator(thickness=BOX_THICKNESS)
trace_annotator     = sv.TraceAnnotator(thickness=BOX_THICKNESS, trace_length=TRACE_LENGTH)

# =============================================================================
#  STATE MACHINE INSTANCE
# =============================================================================

state_machine = TrafficStateMachine(LANE_NAMES)

# =============================================================================
#  HELPER — draw a filled semi-transparent rectangle behind text
# =============================================================================

def draw_panel(img, x, y, w, h, color=(0, 0, 0), alpha=0.5):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# =============================================================================
#  FRAME PROCESSOR
# =============================================================================

vehicle_counts = {cid: 0 for cid in VEHICLE_CLASSES}

def process_frame(frame: np.ndarray, now: float) -> np.ndarray:
    global vehicle_counts
    vehicle_counts = {cid: 0 for cid in VEHICLE_CLASSES}

    # ── Detection ──────────────────────────────────────────────────────────
    results    = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]
    detections = byte_tracker.update_with_detections(detections)

    # ── Feature 5: stopped vs moving ───────────────────────────────────────
    stopped_map: dict[int, bool] = {}
    if len(detections) > 0 and detections.tracker_id is not None:
        stopped_map = motion_tracker.update(
            detections.tracker_id, detections.xyxy
        )

    # ── Count vehicles in this frame (whole frame = current lane) ──────────
    raw_count = len(detections)
    tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
    stopped_count = sum(1 for tid in tracker_ids if stopped_map.get(int(tid), False))

    # ── Feature 1: rolling average ─────────────────────────────────────────
    smoothed_count = lane_smoother.update(raw_count)

    # ── Feature 4: state machine update ────────────────────────────────────
    state_machine.update(smoothed_count, stopped_count, now)

    # ── Global per-class counts ────────────────────────────────────────────
    labels = []
    for conf, class_id, track_id in zip(
        detections.confidence,
        detections.class_id,
        detections.tracker_id if detections.tracker_id is not None
        else [None] * len(detections)
    ):
        vehicle_counts[class_id] += 1
        is_stopped = stopped_map.get(int(track_id) if track_id else -1, False)
        status = 'STOP' if is_stopped else 'MOVE'
        labels.append(f"#{track_id} {CLASS_NAMES[class_id]} {conf:.2f} [{status}]")

    # ── Draw traces and boxes ───────────────────────────────────────────────
    annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    # ── Labels above boxes (red=stopped, cyan=moving) ──────────────────────
    for box, label, tid in zip(
        detections.xyxy,
        labels,
        detections.tracker_id if detections.tracker_id is not None else []
    ):
        x1, y1, _, _ = box.astype(int)
        colour = (0, 0, 255) if stopped_map.get(int(tid), False) else (0, 255, 255)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    # ── Draw frame border colour based on signal state ─────────────────────
    green_lane = state_machine.green_lane
    sig_state  = state_machine.state

    if sig_state == SignalState.GREEN:
        border_color = (0, 220, 0)      # green
    else:
        border_color = (0, 180, 255)    # yellow/amber

    cv2.rectangle(annotated, (4, 4), (FRAME_WIDTH - 4, FRAME_HEIGHT - 4),
                  border_color, 6)

    # ── Line crossing counter ───────────────────────────────────────────────
    line_counter.trigger(detections)
    annotated = line_zone_annotator.annotate(annotated, line_counter=line_counter)

    # ── HUD panel — top-left (vehicle type counts) ─────────────────────────
    total = sum(vehicle_counts.values())
    panel_lines = [f"{CLASS_NAMES[cid]}: {vehicle_counts[cid]}"
                   for cid in VEHICLE_CLASSES]
    panel_lines.append(f"Total (frame): {total}")
    panel_lines.append(f"Stopped:       {stopped_count}")
    panel_lines.append(f"Crossed IN:  {line_counter.in_count}")
    panel_lines.append(f"Crossed OUT: {line_counter.out_count}")

    draw_panel(annotated, 5, 5, 240, len(panel_lines) * 26 + 10)
    for i, txt in enumerate(panel_lines):
        color = (0, 200, 255) if 'Total' in txt else \
                (0, 0, 255)   if 'Stopped' in txt else \
                (255, 200, 0) if 'Crossed' in txt else (0, 255, 0)
        cv2.putText(annotated, txt, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── Current lane info — centre overlay ─────────────────────────────────
    sig_color = (0, 220, 0) if sig_state == SignalState.GREEN else (0, 180, 255)
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT - 60

    draw_panel(annotated, cx - 220, cy - 30, 440, 50, alpha=0.55)
    cv2.putText(annotated,
                f"{green_lane}  |  {sig_state.name}  |  Avg: {smoothed_count:.1f}  Stop: {stopped_count}",
                (cx - 210, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, sig_color, 2)

    # ── JAM / LOW-TRAFFIC early-switch countdown warning ───────────────────
    early_cd = state_machine.early_countdown
    if early_cd is not None and sig_state == SignalState.GREEN:
        warn_color = (0, 80, 255)   # orange-red
        warn_text  = f"  SWITCHING IN {early_cd:.1f}s  "
        (tw, th), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        wx = (FRAME_WIDTH - tw) // 2
        wy = FRAME_HEIGHT // 2 - 50
        draw_panel(annotated, wx - 10, wy - th - 10, tw + 20, th + 20,
                   color=(0, 0, 180), alpha=0.65)
        cv2.putText(annotated, warn_text, (wx, wy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, warn_color, 3)

    # ── Signal status panel — top-right ────────────────────────────────────
    countdown   = state_machine.countdown_to_switch(now)
    last_counts = state_machine.last_counts()
    reason      = state_machine.switch_reason

    stopped_pct = int(100 * stopped_count / max(raw_count, 1))

    sig_lines = [
        f"GREEN LANE: {green_lane}",
        f"State: {sig_state.name}",
        f"Switch in: {countdown:.1f}s",
        f"Min: {MIN_GREEN_SECONDS:.0f}s  Max: {MAX_GREEN_SECONDS:.0f}s",
        f"Jam trigger: >={int(JAM_STOPPED_RATIO*100)}% stopped for {JAM_SECONDS:.0f}s",
        f"Stopped now: {stopped_count}/{raw_count} ({stopped_pct}%)",
        f"Last reason: {reason if reason else '-'}",
        "",  # divider
    ] + [f"{name}: {last_counts[name]:.1f} veh" for name in LANE_NAMES]

    panel_w = 360
    draw_panel(annotated, FRAME_WIDTH - panel_w - 5, 5, panel_w,
               len(sig_lines) * 24 + 10, color=(0, 40, 0))
    for i, txt in enumerate(sig_lines):
        if not txt:
            continue
        is_active_lane = txt.startswith(green_lane)
        is_warn = "Stopped now" in txt and stopped_pct >= int(JAM_STOPPED_RATIO * 100)
        row_color = (0, 80, 255) if is_warn else                     sig_color     if is_active_lane else (180, 180, 180)
        cv2.putText(annotated, txt,
                    (FRAME_WIDTH - panel_w, 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, row_color, 1)

    return annotated

# =============================================================================
#  MAIN LOOP
# =============================================================================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {VIDEO_PATH}")
        return

    print("Stream opened. Press 'q' to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Warning: Frame read failed. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                print("Reconnect failed. Exiting.")
                break
            continue

        output = process_frame(frame, time.time())
        cv2.imshow(WINDOW_TITLE, output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
