"""Métriques communes à tous les modèles (exigences 3, 4, 9)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, accuracy_score)


def best_f1_threshold(y, proba) -> float:
    """Seuil maximisant le F1 — à appeler sur la VALIDATION uniquement."""
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def ece(y, proba, n_bins: int = 10) -> float:
    """Expected Calibration Error, bacs de largeur égale."""
    y, proba = np.asarray(y), np.asarray(proba)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(proba[mask].mean() - y[mask].mean())
    return float(total)


def compute_metrics(y, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "ece": round(ece(y, proba), 4),
        "threshold": round(float(threshold), 4),
    }
