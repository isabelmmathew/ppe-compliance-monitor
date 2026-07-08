from config import (MODEL_PATH, VIDEO_PATH, OUTPUT_PATH, TRACKER_CFG, FRAME_SIZE, CONF_THRESH, ALERT_WINDOW, ALERT_RATIO, CLASS)

from visualisation import(draw_box, draw_hud, stack_images)

from compliance import ComplianceMonitor

from ultralytics import YOLO
import cv2
import time

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError("Cannot open video.")

fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
out_size = (FRAME_SIZE[0] * 2, FRAME_SIZE[1])

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, out_size)

t_prev = time.time()

monitor = ComplianceMonitor(ALERT_WINDOW, ALERT_RATIO)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, FRAME_SIZE)
    annotated = img.copy()

    results = model.track(img, persist=True, tracker=TRACKER_CFG, conf=CONF_THRESH)

    boxes = results[0].boxes

    people, helmets, vests = [], [], []
    seen_ids = set()

    for box in boxes:
        cls = int(box.cls)
        if cls == CLASS["person"] and box.id is not None:
            people.append((int(box.id), tuple(map(int, box.xyxy[0]))))
        elif cls == CLASS["helmet"]:
            helmets.append(tuple(map(int, box.xyxy[0])))
        elif cls == CLASS["vest"]:
            vests.append(tuple(map(int, box.xyxy[0])))

    for h in helmets:
        draw_box(annotated,h[:2], h[2:], (30,30,30))

    for v in vests:
        draw_box(annotated,v[:2], v[2:], (200,100,0))

    total_workers = len(people)
    n_compliant = 0
    n_violations = 0

    for track_id, pbox in people:
        seen_ids.add(track_id)

        compliant, missing = monitor.check_worker(track_id, pbox, helmets, vests)

        colour = (0,200,60) if compliant else (0,0,220)

        if compliant:
            label = f"ID {track_id}"
            n_compliant += 1
        else:
            label = f"ID {track_id} | {', '.join(missing)}"
            n_violations += 1

        x1, y1, x2, y2 = pbox
        draw_box(annotated, (x1,y1), (x2,y2), colour, label)

    monitor.remove_old_tracks(seen_ids)

    draw_hud(annotated, total_workers, n_compliant, n_violations)

    combined = stack_images(0.7, [img, annotated])
    writer.write(cv2.resize(combined, out_size))
    cv2.imshow("PPE Compliance", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Done.")
