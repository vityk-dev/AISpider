from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    assets: Path
    data_processed: Path
    models: Path
    reports: Path

    @staticmethod
    def from_repo(file: str | Path) -> "Paths":
        # scripts/*.py -> repo root is parents[1]
        root = Path(file).resolve().parents[1]
        return Paths(
            root=root,
            assets=root / "assets",
            data_processed=root / "data" / "processed",
            models=root / "models",
            reports=root / "reports",
        )


# --- Gesture labels (order = model output index) ---
LABELS_ORDER = ["FIST", "OPEN_PALM", "POINT_LEFT", "POINT_RIGHT", "THUMB_DOWN", "THUMB_UP"]

# Key mapping for collection / online eval
KEY_TO_LABEL = {
    ord("1"): "OPEN_PALM",
    ord("2"): "FIST",
    ord("3"): "THUMB_UP",
    ord("4"): "THUMB_DOWN",
    ord("5"): "POINT_LEFT",
    ord("6"): "POINT_RIGHT",
}

# Feature settings
USE_Z = False  # keep False (42 features) unless you retrain everything with z

# Default filenames (single source of truth)
CSV_DEFAULT = "gestures.csv"
MODEL_PT = "gesture_mlp.pt"
MODEL_ONNX = "gesture_mlp.onnx"
SCALER_JSON = "scaler.json"
LABELS_JSON = "labels.json"
VIDEO_DEFAULT = "test.mp4"