from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(r"C:\Users\romme\PycharmProjects\YOLO-Vision-Lab\car_highway.mp4")
counted_ids = set()
before_line = set()
after_line = set()
line_height = 300
margin_y = 30
margin_x = 1290
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True, classes=[2])

    cv2.line(frame, (0,line_height), (1445, line_height), (255,0,0))
    cv2.line(frame, (margin_x,0), (margin_x, 650), (255,0,0))

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        counted_ids.update(track_ids)

        for tid, box in zip(track_ids, boxes):
            x1, y1, x2, y2 = box
            center_y = (y2+y1) /2
            center_x = (x1+x2)/2
            if center_y > line_height+margin_y and center_x < margin_x:
                after_line.add(tid)
            elif center_y < line_height-margin_y and center_x < margin_x:
                before_line.add(tid)
            common = before_line & after_line
            count = len(common)
            cv2.putText(frame, str(count), (20,20), cv2.FONT_HERSHEY_SIMPLEX, color=(0,255,0), thickness=2, fontScale=1)
            cv2.putText(frame, str(common), (20,50), cv2.FONT_HERSHEY_SIMPLEX, color=(0,255,0), thickness=2, fontScale=1)

            cv2.putText(frame, str(tid), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
# print(counted_ids)
print(before_line)
print(after_line)
final_set = set()
common = before_line & after_line
print(common)
print(len(common))

cap.release()
cv2.destroyAllWindows()