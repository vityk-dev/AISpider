from __future__ import annotations
import numpy as np


def normalize_landmarks(lm_xyz: np.ndarray) -> np.ndarray:
    """
    lm_xyz shape (21,3) in MediaPipe normalized coords.
    Normalize for translation + scale:
      - translate wrist to origin
      - scale by wrist->middle_mcp distance (9)
    """
    origin = lm_xyz[0].copy()
    pts = lm_xyz - origin

    scale = float(np.linalg.norm(pts[9, :2]))
    if scale < 1e-6:
        scale = 1.0

    pts[:, :2] /= scale
    pts[:, 2] /= scale
    pts = np.clip(pts, -3.0, 3.0)
    return pts


def landmarks_to_features(lm_xyz: np.ndarray, use_z: bool) -> np.ndarray:
    lm_xyz = normalize_landmarks(lm_xyz)
    return lm_xyz.reshape(-1) if use_z else lm_xyz[:, :2].reshape(-1)