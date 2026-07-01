from ultralytics import YOLO
import supervision as sv 
import numpy as np
import cv2
import matplotlib.pyplot as plt

video_path = 'http://10.108.38.126:8080/video'

#loading model
model = YOLO('yolov8x.pt').to('cuda')
model.fuse() #for faster inference

#dictionary mapping
CLASS_NAMES_DICT = model.model.names
CLASS_NAMES_DICT

#class ids of intrest - car, motorcycle, bus,truck

classes = [2,3,5,7]

##plot frame from video
#create frame generator
generator = sv.get_video_frames_generator('car_video5.mp4')

#get first frame
iterator = iter(generator)
frame = next(iterator)
#sv.plot_image(frame,(10,10))

#detect vehicle from frame
results = model(frame,verbose =False)[0]

#convert to detections
detections = sv.Detections.from_ultralytics(results)

#only consider our required classes

detections = detections[np.isin(detections.class_id,classes)]

#create instance for box annotator
box_annotator = sv.BoxAnnotator(thickness=3)
#annotator in frame
annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)

#format labels
labels=[]
for confidence, class_id in zip(detections.confidence, detections.class_id):
    label = f'{CLASS_NAMES_DICT[class_id]}{confidence:0.2f}'
    labels.append(label)

#overlay labels in the bounding box
for box, label in zip(detections.xyxy, labels):
    x1,y1,x2,y2 = box.astype(int)

    #add the label above the box
    cv2.putText(
        annotated_frame, label, (x1, y1-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.2,
        color = (0,255,255), thickness=6
        )

#plot image
# sv.plot_image(annotated_frame, (10,10))

##track and count vehicles
print(sv.VideoInfo.from_video_path(video_path))

#line config

LINE_START = sv.Point(0,500)
LINE_END= sv.Point(1200,500)

#create bytetracker instance
byte_tracker = sv.ByteTrack(frame_rate=25)


#create linezone counter instance
line_counter = sv.LineZone(start=LINE_START, end = LINE_END)

#create linezone annotator
line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

#create box annotator
box_annotator=sv.BoxAnnotator(thickness=2)

#create trace annotator
trace_annotator = sv.TraceAnnotator(thickness=2,trace_length = 60)

#define function for processing frames
def process_frame(frame):
    #get results from model
    results = model(frame, verbose=False)[0]
    #convert to detections
    detections = sv.Detections.from_ultralytics(results)
    
    #only consider our required classes
    detections = detections[np.isin(detections.class_id,classes)]

    #tracking detections
    detections = byte_tracker.update_with_detections(detections)

    #creae labels
    labels=[]
    for confidence, class_id, tracker_id in zip(detections.confidence,detections.class_id, detections.tracker_id):
        label = f'{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}'
        labels.append(label)

    #update trace annotator
    annotated_frame = trace_annotator.annotate(scene = frame.copy(),detections=detections)
    #update box annotator
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    
    #overlay labels in the bounding box
    for box, label in zip(detections.xyxy, labels):
        x1,y1,x2,y2 = box.astype(int)

        #add the label above the box
        cv2.putText(
            annotated_frame, label, (x1, y1-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale = 0.4,
            color = (0,255,255), thickness=1
            )
    line_counter.trigger(detections)
    #update line zone annotator
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_counter)
    return annotated_frame
'''
#run the program
#get input from webcam
# video_cap = cv2.VideoCapture(0)  # capture from webcam

#get input from video
video_cap = cv2.VideoCapture(video_path)

while True:
    success, frame = video_cap.read()
    if not success:
        print('nah')
        break

    #resize frame
    #frame = cv2.resize(frame,(1280,720))

    #convert to RGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #process frame
    processed_frame = process_frame(frame)

    #display frame
    cv2.imshow("Vehicle Tracking and Counting", processed_frame)

    #exit if any key pressed
    if cv2.waitKey(1) & 0xFF !=255:
        break

video_cap.release()
cv2.destroyAllWindows()'''

# keep your existing imports and setup

# class ids of interest
classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# create a dictionary to hold counts
vehicle_counts = {2: 0, 3: 0, 5: 0, 7: 0}

def process_frame(frame):
    global vehicle_counts

    # run YOLO
    results = model(frame, verbose=False)[0]

    # convert to detections
    detections = sv.Detections.from_ultralytics(results)

    # filter classes of interest
    detections = detections[np.isin(detections.class_id, classes)]

    # tracking
    detections = byte_tracker.update_with_detections(detections)

    # reset counts for this frame
    vehicle_counts = {2: 0, 3: 0, 5: 0, 7: 0}

    # create labels
    labels = []
    for confidence, class_id, tracker_id in zip(
        detections.confidence, detections.class_id, detections.tracker_id
    ):
        label = f'{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}'
        labels.append(label)

        # increment class count
        if class_id in vehicle_counts:
            vehicle_counts[class_id] += 1

    # trace annotator
    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)

    # box annotator
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

    # overlay labels
    for box, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.4,
            color=(0, 255, 255),
            thickness=1
        )

    # line counter
    line_counter.trigger(detections)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_counter)

    # draw live counters in top-left
    y_offset = 30
    for cid in vehicle_counts:
        text = f"{CLASS_NAMES_DICT[cid]}: {vehicle_counts[cid]}"
        cv2.putText(
            annotated_frame,
            text,
            (10, y_offset),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=2
        )
        y_offset += 30

    return annotated_frame

    
# ---- main loop ----
# capture from phone camera or video file
video_cap = cv2.VideoCapture(video_path)

while True:
    success, frame = video_cap.read()
    if not success:

        print('nah')
        break

    processed_frame = process_frame(frame)

    cv2.imshow("Vehicle Tracking and Counting", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()

