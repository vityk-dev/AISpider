from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def apply_scaler(feat: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (feat - mean) / (scale + 1e-9)