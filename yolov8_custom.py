'''from ultralytics import YOLO

model = YOLO('pothole.pt')  # Load a pretrained model (you can specify a custom path if needed)

results = model(source=0, show=True, conf=0.4, save=True)  # Run inference on webcam (source=1) with a confidence threshold of 0.4, show results, and save them'''

from ultralytics import YOLO
import cv2

# ============================= CONFIG =============================

VIDEO_PATH   = 'http://10.28.139.225:8080/video'  # IP Webcam URL
MODEL_PATH   = 'pothole.pt'
WINDOW_TITLE = 'Pothole Detection'
CONFIDENCE   = 0.4

# ============================ MODEL SETUP =========================

model = YOLO(MODEL_PATH)

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

        # Run inference on the current frame
        results = model(frame, conf=CONFIDENCE, verbose=False)

        # Annotate frame with detections
        annotated = results[0].plot()

        cv2.imshow(WINDOW_TITLE, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================== ENTRY POINT ===========================

if __name__ == "__main__":
    main()