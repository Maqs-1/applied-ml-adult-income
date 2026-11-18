import numpy as np
import pytest
from src.metrics import best_f1_threshold, compute_metrics


def test_compute_metrics_perfect_predictions():
    """Des probabilités parfaites doivent donner PR-AUC = 1.0, ROC-AUC = 1.0 et Brier = 0.0."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    metrics = compute_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.0)


def test_best_f1_threshold_case():
    """Vérifie que best_f1_threshold s'exécute correctement sur un cas jouet."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_proba = np.array([0.05, 0.10, 0.20, 0.25, 0.35, 0.80])

    thresh = best_f1_threshold(y_true, y_proba)
    assert isinstance(thresh, float)
    assert 0.0 <= thresh <= 1.0
