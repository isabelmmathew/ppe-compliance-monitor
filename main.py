'''
for each detected person:
estimate the head region as the top quarter of the person's bounding box.
determine helmet and vest association using intersection over union (iou) between the detected ppe and person's bounding box.
maintain a short history of helmet and vest detections for each tracked worker using a fixed-size queue.
classify a worker as non-compliant only if the proportion of missing ppe detections within the history window exceeds the specified threshold.

limitations
a single helmet or vest can still be associated with multiple workers.
the head region is approximated as the top 25% of the person's bounding box,
ppe association is based on simple iou thresholds, which may fail when workers overlap or are partially occluded.
the system assumes that all detected people require the same ppe and does not distinguish between different worker roles or work zones.
'''
from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def helmet_overlaps_person(helmet_box, person_box):
    px1, py1, px2, py2 = person_box
    top_box = (px1, py1, px2, py1 + (py2 - py1) / 4)
    return iou(helmet_box, top_box) > 0.05

def vest_overlaps_person(vest_box, person_box):
    return iou(vest_box, person_box) > 0.05

def draw_box(frame, pt1, pt2, colour, label=None):
    cv2.rectangle(frame, pt1, pt2, colour, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = pt1[0], pt1[1] - 6
        cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), colour, -1)
        cv2.putText(frame, label, (lx + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

def draw_hud(frame, total, compliant, violations, fps):
    lines = [
        f"FPS: {fps:.1f}",
        f"Workers: {total}",
        f"Compliant: {compliant}",
        f"Violations: {violations}",
    ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += 20

def stack_images(scale, imgs):
    resized = []
    h, w = imgs[0].shape[:2]
    for img in imgs:
        img = cv2.resize(img, (w, h))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized.append(img)
    combined = np.hstack(resized)
    return cv2.resize(combined, (0,0), fx=scale, fy=scale)

model = YOLO("best.pt")
cap = cv2.VideoCapture("val3.mp4")

if not cap.isOpened():
    raise FileNotFoundError("Cannot open video.")

fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
frame_size = (640, 640)
out_size = (frame_size[0] * 2, frame_size[1])

writer = cv2.VideoWriter(
    "output_annotated.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_src,
    out_size
)

tracker_cfg = "bytetrack.yaml"
conf_thresh = 0.6
alert_window = 15
alert_ratio = 0.6

CLASS = {"helmet":0, "vest":4, "person":3}

worker_history = {}
t_prev = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, frame_size)
    annotated = img.copy()

    results = model.track(
        img,
        persist=True,
        tracker=tracker_cfg,
        conf=conf_thresh
    )

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
        cv2.rectangle(annotated, h[:2], h[2:], (30,30,30), 1)

    for v in vests:
        cv2.rectangle(annotated, v[:2], v[2:], (200,100,0), 1)

    total_workers = len(people)
    n_compliant = 0
    n_violations = 0

    for track_id, pbox in people:
        seen_ids.add(track_id)

        if track_id not in worker_history:
            worker_history[track_id] = {
                "helmet": deque(maxlen=alert_window),
                "vest": deque(maxlen=alert_window)
            }

        helmet_found = any(helmet_overlaps_person(h, pbox) for h in helmets)
        vest_found = any(vest_overlaps_person(v, pbox) for v in vests)

        hist = worker_history[track_id]
        hist["helmet"].append(0 if helmet_found else 1)
        hist["vest"].append(0 if vest_found else 1)

        helmet_alert = (sum(hist["helmet"]) / len(hist["helmet"])) >= alert_ratio if hist["helmet"] else False
        vest_alert = (sum(hist["vest"]) / len(hist["vest"])) >= alert_ratio if hist["vest"] else False

        missing = []
        if helmet_alert:
            missing.append("no helmet")
        if vest_alert:
            missing.append("no vest")

        compliant = not (helmet_alert or vest_alert)
        colour = (0,200,60) if compliant else (0,0,220)

        if compliant:
            label = f"ID {track_id}"
            n_compliant += 1
        else:
            label = f"ID {track_id} | {', '.join(missing)}"
            n_violations += 1

        x1, y1, x2, y2 = pbox
        draw_box(annotated, (x1,y1), (x2,y2), colour, label)

    for gone_id in set(worker_history) - seen_ids:
        del worker_history[gone_id]

    t_now = time.time()
    fps = 1.0 / (t_now - t_prev + 1e-9)
    t_prev = t_now

    draw_hud(annotated, total_workers, n_compliant, n_violations, fps)

    combined = stack_images(0.7, [img, annotated])
    writer.write(cv2.resize(combined, out_size))
    cv2.imshow("PPE Compliance", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Done.")
