from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import torch

# ============================= CONFIG =============================

VIDEO_PATH    = 'http://10.108.38.126:8080/video'
MODEL_PATH    = 'yolov8x.pt'
WINDOW_TITLE  = 'Vehicle Tracking and Counting'

# YOLO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

TRACE_LENGTH  = 60   # frames to keep motion trail
BOX_THICKNESS = 2

# ============================ MODEL SETUP =========================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

model = YOLO(MODEL_PATH).to(DEVICE)
model.fuse()  # fuse layers for faster inference

CLASS_NAMES = model.model.names

# ====================== DYNAMIC VIDEO RESOLUTION =================

def get_video_properties(path):
    """Open the video source once to read resolution and FPS."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fps    = fps if fps and fps > 0 else 25.0  # fallback to 25 if unreadable
    cap.release()
    return width, height, fps

FRAME_WIDTH, FRAME_HEIGHT, FPS = get_video_properties(VIDEO_PATH)
print(f"Source: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS:.1f} fps")

# ==================== TRACKING & ANNOTATION SETUP =================

# Counting line spans full width at vertical midpoint
LINE_START = sv.Point(0, FRAME_HEIGHT // 2)
LINE_END   = sv.Point(FRAME_WIDTH, FRAME_HEIGHT // 2)

byte_tracker       = sv.ByteTrack(frame_rate=int(FPS))
line_counter       = sv.LineZone(start=LINE_START, end=LINE_END)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=2, text_thickness=2, text_scale=1
)
box_annotator   = sv.BoxAnnotator(thickness=BOX_THICKNESS)
trace_annotator = sv.TraceAnnotator(thickness=BOX_THICKNESS, trace_length=TRACE_LENGTH)

# ========================= FRAME PROCESSOR ========================

# Per-frame counts (reset each frame) and cumulative crossed counts
vehicle_counts = {cid: 0 for cid in VEHICLE_CLASSES}

def process_frame(frame):
    """Detect, track, count, and annotate vehicles in a single frame."""
    global vehicle_counts

    # Reset per-frame counts
    vehicle_counts = {cid: 0 for cid in VEHICLE_CLASSES}

    # --- Detection ---
    results    = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

    # --- Tracking ---
    detections = byte_tracker.update_with_detections(detections)

    # --- Labels & counts ---
    labels = []
    for conf, class_id, track_id in zip(
        detections.confidence,
        detections.class_id,
        detections.tracker_id
    ):
        labels.append(f"#{track_id} {CLASS_NAMES[class_id]} {conf:.2f}")
        vehicle_counts[class_id] += 1

    # --- Annotation ---
    annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    # Labels above bounding boxes
    for box, label in zip(detections.xyxy, labels):
        x1, y1, _, _ = box.astype(int)
        cv2.putText(
            annotated, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0, 255, 255),
            thickness=1
        )

    # --- Line crossing counter ---
    line_counter.trigger(detections)
    annotated = line_zone_annotator.annotate(
        annotated, line_counter=line_counter
    )

    # --- Live per-class counter (top-left) ---
    y = 30
    total = 0
    for cid in VEHICLE_CLASSES:
        count = vehicle_counts[cid]
        total += count
        cv2.putText(
            annotated,
            f"{CLASS_NAMES[cid]}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=2
        )
        y += 30

    # Total vehicle count
    cv2.putText(
        annotated,
        f"Total: {total}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0, 200, 255),
        thickness=2
    )

    # Crossed line counts (in/out) — bottom-left
    crossed_in  = line_counter.in_count
    crossed_out = line_counter.out_count
    h = FRAME_HEIGHT
    cv2.putText(annotated, f"Crossed IN:  {crossed_in}",  (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    cv2.putText(annotated, f"Crossed OUT: {crossed_out}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    return annotated

# ============================= MAIN ===============================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Unable to open video source: {VIDEO_PATH}")
        return

    print("Stream opened. Press 'q' to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Warning: Failed to read frame. Attempting to reconnect...")
            cap.release()
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                print("Reconnect failed. Exiting.")
                break
            continue

        output = process_frame(frame)
        cv2.imshow(WINDOW_TITLE, output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================== ENTRY POINT ===========================

if __name__ == "__main__":
    main()
