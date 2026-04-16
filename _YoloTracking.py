from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("car_highway.mp4")
counted_ids = set()
# cap.set(cv2.CAP_PROP_POS_FRAMES, 99)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True, classes=[2])

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        # print(f"track id is {track_ids} ########")
        counted_ids.update(track_ids)

        for tid, box in zip(track_ids, boxes):
            x1, y1, x2, y2 = box
            cv2.putText(frame, str(tid), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
print(counted_ids)
print(len(counted_ids))


cap.release()
cv2.destroyAllWindows()