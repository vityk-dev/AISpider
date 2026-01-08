from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from gestures.config import Paths, CSV_DEFAULT, MODEL_PT, SCALER_JSON, LABELS_JSON
from gestures.mlp import MLP
from gestures.io import save_json


EPOCHS = 30
BATCH = 256
LR = 1e-3


def main():
    paths = Paths.from_repo(__file__)

    csv_path = paths.data_processed / CSV_DEFAULT
    model_out = paths.models / MODEL_PT
    scaler_out = paths.models / SCALER_JSON
    labels_out = paths.models / LABELS_JSON

    if not csv_path.exists():
        raise RuntimeError(f"Missing dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise RuntimeError(f"Dataset is empty: {csv_path}")

    labels = sorted(df["label"].unique().tolist())
    label_to_id = {l: i for i, l in enumerate(labels)}
    y = df["label"].map(label_to_id).to_numpy(dtype=np.int64)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    save_json(scaler_out, {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    save_json(labels_out, labels)

    model = MLP(in_dim=X.shape[1], num_classes=len(labels), dropout=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH, shuffle=True
    )

    val_X = torch.tensor(X_val)
    val_y = torch.tensor(y_val)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_X)
            pred = logits.argmax(dim=1)
            acc = (pred == val_y).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d}/{EPOCHS}  val_acc={acc:.3f}  best={best_acc:.3f}")

    paths.models.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(), model_out)

    print("Saved:")
    print(" ", model_out)
    print(" ", scaler_out)
    print(" ", labels_out)


if __name__ == "__main__":
    main()