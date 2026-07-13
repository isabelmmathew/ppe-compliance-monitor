import csv
import os
import cv2
import shutil


class ViolationLogger:

    def __init__(self, snapshot_dir="snapshots"):
        self.records = []

        # Keeps track of currently active violations
        # Key: (track_id, violation)
        self.active = set()

        self.snapshot_dir = snapshot_dir

        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)

        os.makedirs(snapshot_dir)

        self.snapshot_count = 1

    def log(
        self,
        track_id,
        missing,
        person_box,
        frame,
        frame_number,
        fps):

        if not missing:
            return

        x1, y1, x2, y2 = person_box

        time_seconds = round(frame_number / fps, 2)

        # Create a unique key for the entire violation state
        key = (track_id, tuple(sorted(missing)))

        if key in self.active:
            return

        # Remove any previous active violation for this worker
        self.active = {
            k for k in self.active
            if k[0] != track_id
        }

        self.active.add(key)

        snapshot_name = (
            f"track{track_id}_"
            f"{time_seconds:.2f}s_"
            f"{self.snapshot_count}.jpg"
        )

        snapshot_path = os.path.join(
            self.snapshot_dir,
            snapshot_name
        )

        crop = frame[y1:y2, x1:x2]

        cv2.imwrite(snapshot_path, crop)

        self.records.append({
            "time": time_seconds,
            "track_id": track_id,
            "violation": ", ".join(f"Missing {v}" for v in missing),
            "snapshot": snapshot_name
        })

        self.snapshot_count += 1
        
        
    def clear_resolved(self, track_id, current_missing):
    
        if current_missing:
            return

        self.active = {key for key in self.active if key[0] != track_id}

    def save(self, csv_path):

        with open(csv_path,
                  "w",
                  newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time",
                    "track_id",
                    "violation",
                    "snapshot"
                ]
            )

            writer.writeheader()

            writer.writerows(self.records)