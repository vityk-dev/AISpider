from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import csv
import time
import cv2
import mediapipe as mp
import numpy as np

from gestures.config import Paths, KEY_TO_LABEL, USE_Z, CSV_DEFAULT
from gestures.features import landmarks_to_features


TARGET_FPS = 30
MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6


def main():
    paths = Paths.from_repo(__file__)
    csv_path = paths.data_processed / CSV_DEFAULT
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try VideoCapture(1).")

    mp_hands = mp.solutions.hands

    feat_dim = 63 if USE_Z else 42
    header = ["label"] + [f"f{i}" for i in range(feat_dim)]

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MIN_DET_CONF,
            min_tracking_confidence=MIN_TRK_CONF,
        ) as hands:
            last_time = time.time()
            saved = 0
            current_label = None

            print("Controls:")
            print("  1..6 set active label (see KEY_TO_LABEL in config.py)")
            print("  SPACE = save sample with current label")
            print("  q = quit")
            print(f"Saving to: {csv_path}")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(frame_rgb)

                overlay = frame.copy()
                h, w = overlay.shape[:2]

                lm_feat = None
                if res.multi_hand_landmarks:
                    hand_lms = res.multi_hand_landmarks[0].landmark
                    lm_xyz = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                    lm_feat = landmarks_to_features(lm_xyz, use_z=USE_Z)

                    for i in range(21):
                        x, y = int(hand_lms[i].x * w), int(hand_lms[i].y * h)
                        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)

                cv2.putText(overlay, f"Label: {current_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(overlay, f"Saved: {saved}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                cv2.imshow("Collect Landmarks", overlay)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if key in KEY_TO_LABEL:
                    current_label = KEY_TO_LABEL[key]

                if key == ord(" ") and current_label is not None and lm_feat is not None:
                    writer.writerow([current_label] + lm_feat.astype(float).tolist())
                    saved += 1

                dt = time.time() - last_time
                if dt < 1.0 / TARGET_FPS:
                    time.sleep((1.0 / TARGET_FPS) - dt)
                last_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {csv_path}")


if __name__ == "__main__":
    main()