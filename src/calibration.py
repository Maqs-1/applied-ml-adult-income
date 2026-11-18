"""Exigence 9 — Diagnostic de calibration du modèle final (test).

Courbe de fiabilité + Brier + ECE. On compare aussi à une recalibration isotonique
apprise sur la VALIDATION (le modèle de base est gelé), pour savoir si un correctif simple
suffit avant d'utiliser les probabilités en aval.

Usage : python -m src.calibration
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator

from src.metrics import compute_metrics, ece
from src.utils import load_config, load_split, p


def diagnose(y, proba, n_bins=10) -> dict:
    """Compare probabilité moyenne et fréquence observée par bac de quantiles.

    La courbe trace la probabilité moyenne en abscisse et la part de
    positifs en ordonnée. Sous la diagonale, les scores surestiment la
    fréquence. Des quantiles répétés peuvent réduire le nombre de bacs.
    La moyenne signée des écarts dans le JSON peut masquer des écarts
    opposés qui se compensent.
    """
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    gap = mean_pred - frac_pos  # >0 : prédit plus haut que la réalité -> sur-confiant sur les positifs
    return {"bins_mean_pred": mean_pred.round(4).tolist(), "bins_frac_pos": frac_pos.round(4).tolist(),
            "mean_gap_pred_minus_obs": round(float(gap.mean()), 4),
            "mean_pred": round(float(np.mean(proba)), 4), "observed_rate": round(float(np.mean(y)), 4)}


def main() -> None:
    """Compare sur le test le modèle brut et sa version recalibrée.

    La correction isotonique est apprise sur la validation avec le modèle
    gelé. Le Brier mesure l'erreur quadratique moyenne des probabilités.
    L'ECE agrège les écarts absolus entre score moyen et fréquence
    observée, pondérés par les effectifs de bacs de largeur égale.
    Pour ces deux scores, une valeur plus faible est préférable.
    L'histogramme montre la répartition des probabilités, tandis que la
    courbe de fiabilité compare ces probabilités aux fréquences réelles.
    Les résultats figurent dans runs/calibration.json.
    """
    cfg = load_config()
    final = json.loads(p(cfg["paths"]["results"]).read_text(encoding="utf-8"))["final_model"]
    art = joblib.load(p(f"models/{final}.joblib"))
    pipe = art["pipeline"]
    X_va, y_va = load_split(cfg, "val")
    X_te, y_te = load_split(cfg, "test")

    proba = pipe.predict_proba(X_te)[:, 1]
    iso = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(X_va, y_va)
    proba_iso = iso.predict_proba(X_te)[:, 1]

    out = {"model": final,
           "raw": {"brier": compute_metrics(y_te, proba, art["threshold"])["brier"],
                   "ece": round(ece(y_te, proba), 4), **diagnose(y_te, proba)},
           "isotonic_on_val": {"brier": compute_metrics(y_te, proba_iso, 0.5)["brier"],
                               "ece": round(ece(y_te, proba_iso), 4), **diagnose(y_te, proba_iso)}}

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for lab, pr in [(f"{final} raw", proba), (f"{final} + isotonic", proba_iso)]:
        fp, mp = calibration_curve(y_te, pr, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, "o-", label=f"{lab} (ECE={ece(y_te, pr):.3f})")
    ax.set(xlabel="mean predicted probability", ylabel="observed share of positives",
           title="Reliability curve (test)")
    ax.legend()
    axh.hist(proba, bins=40, alpha=0.6, label="raw")
    axh.hist(proba_iso, bins=40, alpha=0.6, label="isotonic")
    axh.set(title="Distribution of predicted probabilities", xlabel="p(>50K)")
    axh.legend()
    fig.tight_layout()
    fig.savefig(p("figures/calibration_reliability.png"), dpi=120)
    plt.close(fig)

    joblib.dump(iso, p(f"models/{final}_isotonic.joblib"))
    with open(p("runs/calibration.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
