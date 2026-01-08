from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from gestures.config import Paths, USE_Z, MODEL_ONNX, LABELS_JSON, SCALER_JSON, VIDEO_DEFAULT
from gestures.features import landmarks_to_features
from gestures.infer import softmax, apply_scaler
from gestures.io import load_json, load_scaler


# ---------------- Schedule ----------------
SCHEDULE = ["OPEN_PALM", "FIST", "THUMB_UP", "THUMB_DOWN", "POINT_RIGHT", "POINT_LEFT"]

SAMPLE_FPS = 5
MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6

# Search ranges
SEG_MIN, SEG_MAX, SEG_STEP = 2.5, 6.0, 0.1   # seconds
OFF_STEP = 0.1                               # seconds

IGNORE_EDGE_S = 0.7                          # transition buffer
SMOOTH_WINDOW = 1                            # keep 1 usually


def true_label_at(t: float, start_offset: float, seg_len: float):
    t0 = t - start_offset
    if t0 < 0:
        return None
    cycle_len = seg_len * len(SCHEDULE)
    within = t0 % cycle_len
    seg_idx = int(within // seg_len)
    seg_pos = within - seg_idx * seg_len
    if seg_pos < IGNORE_EDGE_S or seg_pos > (seg_len - IGNORE_EDGE_S):
        return None
    return SCHEDULE[seg_idx]


def show_cm(labels, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def main():
    paths = Paths.from_repo(__file__)

    video_path = paths.assets / "demo_videos" / VIDEO_DEFAULT
    model_path = paths.models / MODEL_ONNX
    labels_path = paths.models / LABELS_JSON
    scaler_path = paths.models / SCALER_JSON

    labels = load_json(labels_path)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    mean, scale = load_scaler(scaler_path)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / SAMPLE_FPS)))

    # ---- PASS 1: sample + infer once ----
    times = []
    probs_list = []
    ok_mask = []

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
    ) as hands:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue

            t = idx / video_fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)

            if not res.multi_hand_landmarks:
                times.append(t)
                probs_list.append(None)
                ok_mask.append(False)
                idx += 1
                continue

            hand_lms = res.multi_hand_landmarks[0].landmark
            lm_xyz = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

            feat = landmarks_to_features(lm_xyz, use_z=USE_Z)
            feat = apply_scaler(feat, mean, scale)

            logits = sess.run(None, {in_name: feat.astype(np.float32)[None, :]})[0][0]
            probs = softmax(logits)

            times.append(t)
            probs_list.append(probs)
            ok_mask.append(True)

            idx += 1

    cap.release()
    print(f"Sampled frames: {len(times)}  usable: {sum(ok_mask)}  video_fps={video_fps:.2f} step={step}")

    pred_ids = np.array([int(np.argmax(p)) if p is not None else -1 for p in probs_list], dtype=int)

    # ---- PASS 2: grid search ----
    best = (0.0, None, None, 0)  # acc, seg_len, offset, n

    for seg_len in np.arange(SEG_MIN, SEG_MAX + 1e-9, SEG_STEP):
        cycle_len = seg_len * len(SCHEDULE)
        for offset in np.arange(0.0, cycle_len, OFF_STEP):
            y_true, y_pred = [], []
            for t, pid, usable in zip(times, pred_ids, ok_mask):
                if not usable:
                    continue
                tl = true_label_at(t, offset, seg_len)
                if tl is None:
                    continue
                y_true.append(label_to_idx[tl])
                y_pred.append(pid)

            if len(y_true) < 80:
                continue

            acc = accuracy_score(y_true, y_pred)
            if acc > best[0]:
                best = (float(acc), float(seg_len), float(offset), len(y_true))

    if best[1] is None:
        print("No valid (seg_len, offset) found. Lower IGNORE_EDGE_S or record a more regular schedule.")
        return

    print(f"\nBEST FOUND: acc={best[0]:.3f}  SEG_LEN_S={best[1]:.2f}  START_OFFSET_S={best[2]:.2f}  samples={best[3]}")

    # ---- PASS 3: report ----
    best_seg, best_off = best[1], best[2]

    y_true, y_pred = [], []
    for t, pid, usable in zip(times, pred_ids, ok_mask):
        if not usable:
            continue
        tl = true_label_at(t, best_off, best_seg)
        if tl is None:
            continue
        y_true.append(label_to_idx[tl])
        y_pred.append(pid)

    print("\nFinal classification report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=3, zero_division=0))
    print("Final accuracy:", accuracy_score(y_true, y_pred))

    show_cm(labels, y_true, y_pred)


if __name__ == "__main__":
    main()