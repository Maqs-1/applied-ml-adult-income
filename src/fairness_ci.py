"""Intervalles bootstrap de l'écart d'equalized odds sur le split de test.

Exécuter python -m src.fairness_ci après l'évaluation du modèle final.
"""
from __future__ import annotations

import json

import numpy as np

from src.fairness import evaluation_predictions
from src.utils import load_config, p


def confusion_counts(y, pred, groups):
    """Compte les décisions par groupe, classe réelle et classe prédite."""
    _, group_indices = np.unique(groups, return_inverse=True)
    counts = np.zeros((group_indices.max() + 1, 2, 2), dtype=int)
    np.add.at(counts, (group_indices, np.asarray(y), np.asarray(pred)), 1)
    return counts


def bootstrap_interval(counts, rng, n_resamples=1000):
    """Calcule un IC percentile à 95 % en stratifiant par groupe et classe réelle.

    Dans chaque strate, le nombre de prédictions positives suit une loi
    binomiale équivalente au rééchantillonnage avec remise des lignes.
    L'incertitude est conditionnelle aux effectifs et aux décisions observées.
    """
    totals = counts.sum(axis=2)
    if np.any(totals == 0):
        raise ValueError("Chaque groupe doit contenir les deux classes réelles.")
    rates = counts[:, :, 1] / totals
    draws = rng.binomial(totals, rates, size=(n_resamples, *totals.shape)) / totals
    gaps = np.ptp(draws, axis=1).max(axis=1)
    lower, upper = np.quantile(gaps, [0.025, 0.975])
    return {"estimate": float(np.ptp(rates, axis=0).max()),
            "ci_95": [float(lower), float(upper)]}


def report(model, sensitive, before, after, seed):
    """Produit les intervalles marginaux avant et après mitigation."""
    rng = np.random.default_rng(seed)
    return {
        "model": model,
        "sensitive": sensitive,
        "metric": "equalized_odds_diff",
        "method": "stratified_percentile_bootstrap",
        "n_resamples": 1000,
        "seed": seed,
        "before": bootstrap_interval(before, rng),
        "after_threshold_optimizer": bootstrap_interval(after, rng),
    }


def main():
    cfg = load_config()
    sensitive = cfg["features"]["sensitive"]
    model, X_test, y_test, pred_before, pred_after, _ = evaluation_predictions(cfg)
    groups = X_test[sensitive].to_numpy()
    out = report(model, sensitive, confusion_counts(y_test, pred_before, groups),
                 confusion_counts(y_test, pred_after, groups), cfg["seed"])
    output = json.dumps(out, indent=2)
    p("runs/fairness_ci.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
