from __future__ import annotations
import sys
import time
from pathlib import Path
from collections import deque

# --- 1. SETUP PATHS DYNAMICALLY ---
# Find the 'spider-llm-operator' root folder
# script is in: spider-llm-operator/vision/scripts/03...
# parents[0]=scripts, parents[1]=vision, parents[2]=spider-llm-operator
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add root to sys.path so we can import 'vision.spider_vision'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

# --- 2. UPDATED IMPORTS (The "Pro" Way) ---
# Now we import from the new package name 'vision.spider_vision'
from vision.spider_vision.config import Paths, USE_Z, MODEL_ONNX, LABELS_JSON, SCALER_JSON
from vision.spider_vision.features import landmarks_to_features
from vision.spider_vision.infer import softmax, apply_scaler
from vision.spider_vision.io import load_json, load_scaler

# ... rest of your code ...


# ---------- SETTINGS ----------
MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6

CONF_THRESHOLD = 0.80         # accept gesture only above this confidence
WINDOW = 10                   # smoothing window length
REQUIRED = 7                  # require gesture to appear >= REQUIRED in last WINDOW
COOLDOWN_S = 0.4              # minimum time between command changes


def main():
    paths = Paths.from_repo(__file__)

    model_path = paths.models / MODEL_ONNX
    labels_path = paths.models / LABELS_JSON
    scaler_path = paths.models / SCALER_JSON

    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")
    if not labels_path.exists():
        raise RuntimeError(f"Missing labels: {labels_path}")
    if not scaler_path.exists():
        raise RuntimeError(f"Missing scaler: {scaler_path}")

    labels = load_json(labels_path)
    mean, scale = load_scaler(scaler_path)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try VideoCapture(1).")

    pred_hist = deque(maxlen=WINDOW)
    last_cmd = None
    last_cmd_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)

            display_cmd = "NONE"
            display_conf = 0.0

            if res.multi_hand_landmarks:
                h, w = frame.shape[:2]
                hand_lms = res.multi_hand_landmarks[0].landmark
                lm_xyz = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

                # Draw landmarks (debug)
                for i in range(21):
                    x, y = int(hand_lms[i].x * w), int(hand_lms[i].y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                feat = landmarks_to_features(lm_xyz, use_z=USE_Z)
                feat = apply_scaler(feat, mean, scale)

                logits = sess.run(None, {in_name: feat.astype(np.float32)[None, :]})[0][0]
                probs = softmax(logits)

                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])
                pred_label = labels[pred_id]

                pred_hist.append(pred_label)

                # Majority vote over last WINDOW
                counts = {}
                for p in pred_hist:
                    counts[p] = counts.get(p, 0) + 1
                best_label = max(counts, key=counts.get)
                best_count = counts[best_label]

                display_cmd = best_label
                display_conf = conf

                now = time.time()
                if conf >= CONF_THRESHOLD and best_count >= REQUIRED:
                    if last_cmd != best_label and (now - last_cmd_time) >= COOLDOWN_S:
                        last_cmd = best_label
                        last_cmd_time = now
                        print(f"COMMAND: {last_cmd}  (conf={conf:.2f}, stable={best_count}/{WINDOW})")

            cv2.putText(
                frame, f"Pred: {display_cmd}  conf={display_conf:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"Stable: {last_cmd}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )
            cv2.putText(
                frame, "q: quit",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            cv2.imshow("Realtime Gesture Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()