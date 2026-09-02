"""Reusable classification metrics for thesis experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred) -> dict:
    """Calculate common binary metrics with fake represented by class 1."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    weighted = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        average="weighted",
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1])
    return {
        "rows": int(len(y_true_array)),
        "real_rows": int(np.sum(y_true_array == 0)),
        "fake_rows": int(np.sum(y_true_array == 1)),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "precision_real": float(per_class[0][0]),
        "recall_real": float(per_class[1][0]),
        "f1_real": float(per_class[2][0]),
        "precision_fake": float(per_class[0][1]),
        "recall_fake": float(per_class[1][1]),
        "f1_fake": float(per_class[2][1]),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
    }


def fake_decision_scores(model, values) -> np.ndarray:
    """Return a classifier score oriented toward fake, never a probability."""
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Expected classes 0 and 1, found {classes}")

    decision_function = getattr(model, "decision_function", None)
    if decision_function is not None:
        scores = np.asarray(decision_function(values))
        if scores.ndim == 1:
            if classes[1] != 1:
                scores = -scores
            return scores.astype(float)
        return (scores[:, classes.index(1)] - scores[:, classes.index(0)]).astype(
            float
        )

    predict_log_proba = getattr(model, "predict_log_proba", None)
    if predict_log_proba is None:
        raise TypeError("Classifier has neither decision_function nor predict_log_proba.")
    log_probabilities = np.asarray(predict_log_proba(values), dtype=float)
    return (
        log_probabilities[:, classes.index(1)]
        - log_probabilities[:, classes.index(0)]
    )


def rounded_metrics(metrics: dict) -> dict:
    """Round only floating values for human-facing artifacts."""
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }
