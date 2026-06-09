"""Evaluation metrics: accuracy, macro precision/recall/F1, and MCC."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return the metric suite used throughout the paper.

    Macro averaging is used for precision/recall/F1 so that minority classes
    (e.g. the 25+ age group) are weighted equally, matching the multi-class
    reporting in the baseline.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro",
                                      zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def percentage_difference(n1: float, n2: float) -> float:
    """Symmetric percentage difference used in the baseline paper (Eq. 8)."""
    denom = (n1 + n2) / 2.0
    return 0.0 if denom == 0 else (n1 - n2) / denom * 100.0
