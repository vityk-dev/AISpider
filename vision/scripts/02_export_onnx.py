from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
import json

from gestures.config import Paths, MODEL_PT, MODEL_ONNX, LABELS_JSON
from gestures.mlp import MLP


def main():
    paths = Paths.from_repo(__file__)

    model_pt = paths.models / MODEL_PT
    labels_path = paths.models / LABELS_JSON
    onnx_out = paths.models / MODEL_ONNX

    if not model_pt.exists():
        raise RuntimeError(f"Missing: {model_pt}")
    if not labels_path.exists():
        raise RuntimeError(f"Missing: {labels_path}")

    labels = json.load(open(labels_path, "r"))
    state = torch.load(model_pt, map_location="cpu")

    in_dim = state["net.0.weight"].shape[1]
    model = MLP(in_dim=in_dim, num_classes=len(labels), dropout=0.0)
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        model, dummy, str(onnx_out),
        input_names=["x"], output_names=["logits"],
        opset_version=12
    )

    print("Exported:", onnx_out)


if __name__ == "__main__":
    main()