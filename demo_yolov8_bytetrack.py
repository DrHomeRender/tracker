import cv2
import numpy as np
from ultralytics import YOLO
from byte_tracker import BYTETracker

model = YOLO("yolov8n.pt")
tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8)

cap = cv2.VideoCapture(0)  # or path to video file
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    dets = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
    detections = np.concatenate([dets, confs], axis=1)

    tracks = tracker.update(detections)
    for tid, x1, y1, x2, y2 in tracks:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f'ID {tid}', (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("ByteTrack Demo", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
