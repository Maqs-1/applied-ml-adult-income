"""Exigence 4 — Évaluation finale sur le TEST (touché une seule fois, ici).

Le modèle final est choisi sur la PR-AUC de VALIDATION (runs/train_results.json),
avant de regarder le test. Les 3 modèles sont ensuite rapportés sur le test avec
les mêmes métriques et leur seuil choisi en validation.

Usage : python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

from src.metrics import compute_metrics
from src.utils import load_config, load_split, p


def select_final(train_results: dict, models: list[str]) -> str:
    return max(models, key=lambda m: train_results[m]["val_at_thr"]["pr_auc"])


def main() -> None:
    cfg = load_config()
    models = cfg["models"]
    tr = json.loads(p("runs/train_results.json").read_text(encoding="utf-8"))
    final = select_final(tr, models)
    X_te, y_te = load_split(cfg, "test")

    out = {"final_model": final, "selection_rule": "max PR-AUC en validation", "test": {}}
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in models:
        art = joblib.load(p(f"models/{name}.joblib"))
        proba = art["pipeline"].predict_proba(X_te)[:, 1]
        out["test"][name] = {"at_val_threshold": compute_metrics(y_te, proba, art["threshold"]),
                             "at_0.5": compute_metrics(y_te, proba, 0.5)}
        PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax, name=name)
    ax.axhline(y_te.mean(), ls="--", c="grey", label="baseline (prévalence)")
    ax.set_title("Courbes précision-rappel — test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p("figures/test_pr_curves.png"), dpi=120)
    plt.close(fig)

    with open(p(cfg["paths"]["results"]), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
