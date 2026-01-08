from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(labels, y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title(title)
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