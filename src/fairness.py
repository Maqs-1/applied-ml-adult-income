"""Exigence 10 — Audit d'équité (attribut sensible : sex ; secondaire : race) et mitigation.

Définitions :
  - Parité démographique : P(Ŷ=1 | A=a) égale entre groupes (diff. des taux de sélection).
  - Égalité des chances / equalized odds : TPR et FPR égaux entre groupes.
On retient equalized odds comme cible de mitigation : les taux de base (>50K) diffèrent
réellement entre hommes et femmes dans le recensement de 1994 ; imposer la parité
démographique forcerait le modèle à ignorer ce signal, alors qu'equalized odds demande
que, à revenu réel égal, l'erreur soit la même quel que soit le sexe.

Mitigation : ThresholdOptimizer (post-traitement, fairlearn) appris sur la VALIDATION,
évalué une fois sur le TEST. Le modèle de base est inchangé (prefit).

Usage : python -m src.fairness
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_difference, demographic_parity_ratio,
                               equalized_odds_difference, false_positive_rate, selection_rate,
                               true_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score

from src.utils import load_config, load_split, p

GROUP_METRICS = {"selection_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate,
                 "precision": lambda y, yp: precision_score(y, yp, zero_division=0),
                 "accuracy": accuracy_score, "f1": f1_score, "count": lambda y, yp: len(y)}


class Float64Proba(ClassifierMixin, BaseEstimator):
    """Adaptateur : renvoie des probabilités float64 (XGBoost sort du float32,
    que fairlearn 0.15 + pandas 3 refusent à l'affectation)."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    @property
    def classes_(self):
        return self.pipeline.classes_

    def predict_proba(self, X):
        return np.asarray(self.pipeline.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return self.pipeline.predict(X)


def audit(y, pred, sens) -> dict:
    mf = MetricFrame(metrics=GROUP_METRICS, y_true=y, y_pred=pred, sensitive_features=sens)
    return {"by_group": mf.by_group.round(4).to_dict(orient="index"),
            "overall": mf.overall.round(4).to_dict(),
            "demographic_parity_diff": round(float(demographic_parity_difference(y, pred, sensitive_features=sens)), 4),
            "demographic_parity_ratio": round(float(demographic_parity_ratio(y, pred, sensitive_features=sens)), 4),
            "equalized_odds_diff": round(float(equalized_odds_difference(y, pred, sensitive_features=sens)), 4)}


def evaluation_predictions(cfg):
    """Prépare les décisions avant et après mitigation sur le même test."""
    sens_col = cfg["features"]["sensitive"]
    final = json.loads(p(cfg["paths"]["results"]).read_text(encoding="utf-8"))["final_model"]
    art = joblib.load(p(f"models/{final}.joblib"))
    pipe, thr = art["pipeline"], art["threshold"]
    # index remis à zéro + attribut sensible en numpy : contourne une incompatibilité
    # d'alignement d'index entre fairlearn 0.15 et pandas 3
    X_va, y_va = (o.reset_index(drop=True) for o in load_split(cfg, "val"))
    X_te, y_te = (o.reset_index(drop=True) for o in load_split(cfg, "test"))

    pred_before = (pipe.predict_proba(X_te)[:, 1] >= thr).astype(int)

    to = ThresholdOptimizer(estimator=Float64Proba(pipe), constraints="equalized_odds", objective="balanced_accuracy_score",
                            predict_method="predict_proba", prefit=True)
    to.fit(X_va, y_va, sensitive_features=X_va[sens_col].to_numpy())
    pred_after = to.predict(X_te, sensitive_features=X_te[sens_col].to_numpy(), random_state=cfg["seed"])

    return final, X_te, y_te, pred_before, pred_after, to


def main() -> None:
    cfg = load_config()
    sens_col = cfg["features"]["sensitive"]
    sec_col = cfg["features"]["sensitive_secondary"]
    final, X_te, y_te, pred_before, pred_after, to = evaluation_predictions(cfg)

    out = {"model": final, "sensitive": sens_col, "constraint": "equalized_odds",
           "before": audit(y_te, pred_before, X_te[sens_col]),
           "after_threshold_optimizer": audit(y_te, pred_after, X_te[sens_col]),
           "secondary_race_before": audit(y_te, pred_before, X_te[sec_col])}

    rows = []
    for tag, pr in [("before", pred_before), ("after", pred_after)]:
        mf = MetricFrame(metrics={k: GROUP_METRICS[k] for k in ["selection_rate", "tpr", "fpr"]},
                         y_true=y_te, y_pred=pr, sensitive_features=X_te[sens_col])
        for g, r in mf.by_group.iterrows():
            rows += [{"phase": tag, "group": g, "metric": m, "value": v} for m, v in r.items()]
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, m in zip(axes, ["selection_rate", "tpr", "fpr"]):
        df[df["metric"] == m].pivot(index="phase", columns="group", values="value").loc[["before", "after"]].plot.bar(ax=ax, rot=0)
        ax.set_title(m)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(f"Fairness by {sens_col}: {final}, test (before/after ThresholdOptimizer)")
    fig.tight_layout()
    fig.savefig(p("figures/fairness_before_after.png"), dpi=120)
    plt.close(fig)

    joblib.dump(to, p(f"models/{final}_threshold_optimizer.joblib"))
    with open(p("runs/fairness.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
