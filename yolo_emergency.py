from ultralytics import YOLO

model = YOLO('emergency.pt')  # Load a pretrained model (you can specify a custom path if needed)

results = model(source=0, show=True, conf=0.4, save=True)  # Run inference on webcam (source=1) with a confidence threshold of 0.4, show results, and save them