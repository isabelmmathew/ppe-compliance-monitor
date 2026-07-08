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



class ComplianceMonitor:

    def __init__(self, alert_window, alert_ratio):
        self.alert_window = alert_window
        self.alert_ratio = alert_ratio
        self.worker_history = {}

    #can be extended for more ppe equipment checks
    def check_worker(self, track_id, person_box, helmets, vests,):
        if track_id not in self.worker_history:
            self.worker_history[track_id] = {"helmet": deque(maxlen = self.alert_window), "vest": deque(maxlen=self.alert_window)}

        helmet_found = any(helmet_overlaps_person(h, person_box) for h in helmets)
        vest_found = any(vest_overlaps_person(v, person_box) for v in vests)

        hist = self.worker_history[track_id]
        hist["helmet"].append(0 if helmet_found else 1)
        hist["vest"].append(0 if vest_found else 1)

        helmet_alert = (sum(hist["helmet"]) / len(hist["helmet"])) >= self.alert_ratio 
        vest_alert = (sum(hist["vest"]) / len(hist["vest"])) >= self.alert_ratio 

        missing = []

        if helmet_alert:
            missing.append("no helmet")
        if vest_alert:
            missing.append("no vest")

        compliant = not(helmet_alert or vest_alert)
        label = f"ID: {track_id}"

        if compliant:
            color = (0,255,0)
        else:
            color = (0,0,255)
            label = f"{label} | {','.join(missing)}"

        return (compliant, missing)
    
    def remove_old_tracks(self, seen_ids):
        for gone_id in set(self.worker_history) - seen_ids:
            del self.worker_history[gone_id]
