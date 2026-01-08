from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def save_json(path: Path, obj) -> None:
    """Save JSON with pretty formatting; auto-create parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    """Load JSON."""
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load scaler.json -> (mean, scale) as float32 numpy arrays."""
    s = load_json(path)
    mean = np.array(s["mean"], dtype=np.float32)
    scale = np.array(s["scale"], dtype=np.float32)
    return mean, scale