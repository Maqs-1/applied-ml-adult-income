"""Exigence 8 — Explicabilité SHAP (globale + locale) du modèle final.

Usage : python -m src.explain
Sorties : figures/shap_summary.png, figures/shap_bar.png, figures/shap_local_*.png,
          runs/shap_global.csv, runs/shap_local.json
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import load_config, load_split, p


def explainer_for(model, X_bg):
    name = type(model).__name__
    if name in ("XGBClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, X_bg)


def main(n_sample: int = 1000, n_local: int = 4) -> None:
    cfg = load_config()
    final = json.loads(p(cfg["paths"]["results"]).read_text(encoding="utf-8"))["final_model"]
    art = joblib.load(p(f"models/{final}.joblib"))
    pipe, thr = art["pipeline"], art["threshold"]
    X_te, y_te = load_split(cfg, "test")

    pre = pipe[:-1]
    model = pipe[-1]
    X_sample = X_te.sample(n=min(n_sample, len(X_te)), random_state=cfg["seed"])
    Z = pd.DataFrame(pre.transform(X_sample), columns=pre[-1].get_feature_names_out(), index=X_sample.index)
    explainer = explainer_for(model, Z)
    sv = explainer(Z)
    if sv.values.ndim == 3:  # RF : une sortie par classe -> classe positive
        sv = sv[:, :, 1]

    shap.summary_plot(sv, Z, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(p("figures/shap_summary.png"), dpi=120); plt.close()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(p("figures/shap_bar.png"), dpi=120); plt.close()

    imp = pd.Series(np.abs(sv.values).mean(0), index=Z.columns).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(p("runs/shap_global.csv"))

    # Cas locaux : un vrai positif, un vrai négatif, un faux positif, un faux négatif
    proba = pipe.predict_proba(X_sample)[:, 1]
    pred = (proba >= thr).astype(int)
    y_s = y_te.loc[X_sample.index].values
    cases = {"vrai_positif": (pred == 1) & (y_s == 1), "vrai_negatif": (pred == 0) & (y_s == 0),
             "faux_positif": (pred == 1) & (y_s == 0), "faux_negatif": (pred == 0) & (y_s == 1)}
    local = {}
    for label, mask in list(cases.items())[:n_local]:
        if not mask.any():
            continue
        i = int(np.flatnonzero(mask)[0])
        shap.plots.waterfall(sv[i], max_display=10, show=False)
        plt.tight_layout(); plt.savefig(p(f"figures/shap_local_{label}.png"), dpi=120); plt.close()
        contrib = pd.Series(sv.values[i], index=Z.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:6]
        local[label] = {"row_index": int(X_sample.index[i]), "proba": round(float(proba[i]), 4),
                        "y_true": int(y_s[i]), "raw_features": X_sample.iloc[i].astype(str).to_dict(),
                        "top_contributions": {k: round(float(v), 4) for k, v in top.items()}}
    with open(p("runs/shap_local.json"), "w", encoding="utf-8") as f:
        json.dump({"model": final, "base_value": float(np.ravel(sv.base_values)[0]), "cases": local},
                  f, indent=2, ensure_ascii=False)
    print(imp.head(15))


if __name__ == "__main__":
    main()
