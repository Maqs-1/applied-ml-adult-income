"""Supplementary figures and statistics used to analyse the results in the report.

Does not change any model or decision: it only reloads the saved pipelines and the
splits, draws extra diagnostic plots and writes the numbers quoted in the report to
runs/extra_stats.json (keys are LaTeX-safe macro names).

Usage: python -m src.extra_figures
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
from sklearn.metrics import (RocCurveDisplay, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from src.metrics import ece
from src.utils import load_config, load_split, p

NAMES = {"logreg": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}
S = {}  # statistics quoted in the report


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.1f}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(p(f"figures/{name}"), dpi=130)
    plt.close(fig)


def eda_extra(X, y):
    """Training split only."""
    d = X.copy()
    d["y"] = y.values
    occ = d["occupation"].fillna("Missing")
    rate_occ = d.groupby(occ)["y"].mean().sort_values()
    rel = d.groupby("relationship")["y"].mean().sort_values()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    rate_occ.plot.barh(ax=ax[0], color=["#C44E52" if i == "Missing" else "#4C72B0" for i in rate_occ.index])
    ax[0].axvline(d["y"].mean(), ls="--", c="grey")
    ax[0].set(title="Share >50K by occupation (train)", xlabel="share >50K", ylabel="")
    rel.plot.barh(ax=ax[1], color="#55A868")
    ax[1].axvline(d["y"].mean(), ls="--", c="grey")
    ax[1].set(title="Share >50K by relationship (train)", xlabel="share >50K", ylabel="")
    cg = np.log1p(d["capital-gain"])
    ax[2].hist([cg[d["y"] == 0], cg[d["y"] == 1]], bins=30, label=["<=50K", ">50K"], log=True)
    ax[2].set(title="log(1 + capital gain), log-scaled counts", xlabel="log(1 + capital gain)")
    ax[2].legend()
    save(fig, "eda_extra.png")

    S["OccMissRate"] = pct(rate_occ["Missing"])
    S["OccExecRate"] = pct(rate_occ["Exec-managerial"])
    S["OccProfRate"] = pct(rate_occ["Prof-specialty"])
    S["OccServRate"] = pct(rate_occ["Other-service"])
    S["OccHouseRate"] = pct(rate_occ["Priv-house-serv"])
    S["RelHusbandRate"] = pct(rel["Husband"])
    S["RelWifeRate"] = pct(rel["Wife"])
    S["RelChildRate"] = pct(rel["Own-child"])
    S["TrainPosRate"] = pct(d["y"].mean())
    has = d["capital-gain"] > 0
    S["CapGainPosRate"] = pct(d.loc[has, "y"].mean())
    S["CapGainZeroPosRate"] = pct(d.loc[~has, "y"].mean())
    S["CapGainShare"] = pct(has.mean())
    both = d["workclass"].isna() & d["occupation"].isna()
    S["MissBothShare"] = pct(both.sum() / max(d["occupation"].isna().sum(), 1))
    husband_share_male = (d.loc[d["relationship"] == "Husband", "sex"] == "Male").mean()
    S["HusbandMaleShare"] = pct(husband_share_male)
    married = d["marital-status"].isin(["Married-civ-spouse", "Married-AF-spouse"])
    S["MarriedMaleShare"] = pct((d.loc[married, "sex"] == "Male").mean())
    S["MaleMarriedRate"] = pct(married[d["sex"] == "Male"].mean())
    S["FemaleMarriedRate"] = pct(married[d["sex"] == "Female"].mean())


def model_figures(models, X_va, y_va, X_te, y_te):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    for name, art in models.items():
        proba = art["pipeline"].predict_proba(X_te)[:, 1]
        RocCurveDisplay.from_predictions(y_te, proba, ax=axes[0], name=NAMES[name])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("ROC curves (test)")
    for ax, (name, art) in zip(axes[1:], models.items()):
        proba = art["pipeline"].predict_proba(X_te)[:, 1]
        cm = confusion_matrix(y_te, (proba >= art["threshold"]).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred <=50K", "pred >50K"],
               yticklabels=["true <=50K", "true >50K"], title=f"{NAMES[name]} (thr={art['threshold']:.2f})")
        tn, fp, fn, tp = cm.ravel()
        key = {"logreg": "Log", "random_forest": "Rf", "xgboost": "Xgb"}[name]
        S[f"Tn{key}"], S[f"Fp{key}"], S[f"Fn{key}"], S[f"Tp{key}"] = str(tn), str(fp), str(fn), str(tp)
    save(fig, "test_roc_confusion.png")

    # Threshold sweep on the VALIDATION set for the final model
    art = models["xgboost"]
    proba = art["pipeline"].predict_proba(X_va)[:, 1]
    prec, rec, thr = precision_recall_curve(y_va, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(thr, prec[:-1], label="precision")
    ax.plot(thr, rec[:-1], label="recall")
    ax.plot(thr, f1[:-1], label="F1", lw=2)
    ax.axvline(0.5, c="grey", ls=":", label="0.5")
    ax.axvline(art["threshold"], c="k", ls="--", label=f"chosen = {art['threshold']:.3f}")
    ax.set(xlabel="decision threshold", ylabel="score", title="XGBoost: threshold sweep on the validation set")
    ax.legend(loc="lower left")
    save(fig, "threshold_sweep.png")
    i_best = int(np.argmax(f1[:-1]))
    S["SweepBestFone"] = f3(f1[i_best])
    near = thr[(f1[:-1] >= f1[i_best] - 0.01)]
    S["SweepFlatLo"] = f"{near.min():.2f}"
    S["SweepFlatHi"] = f"{near.max():.2f}"


def optuna_landscape(cfg):
    log = pd.read_csv(p(cfg["paths"]["trials_log"]))
    log = log[log["state"] == "COMPLETE"].copy()
    log["params"] = log["params"].apply(json.loads)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    data = [log.loc[log["model"] == m, "cv_pr_auc_mean"].values for m in NAMES]
    ax[0].boxplot(data, tick_labels=["LogReg", "RF", "XGB"], showfliers=True)
    for i, d in enumerate(data):
        ax[0].scatter(np.random.default_rng(0).normal(i + 1, 0.04, len(d)), d, s=10, alpha=0.5)
    ax[0].set(title="CV PR-AUC of all 40 trials", ylabel="CV PR-AUC")

    lr = log[log["model"] == "logreg"]
    for cw, mk in [(None, "o"), ("balanced", "x")]:
        sub = lr[lr["params"].apply(lambda q: q["class_weight"] == cw)]
        ax[1].scatter(sub["params"].apply(lambda q: q["C"]), sub["cv_pr_auc_mean"], marker=mk, label=f"class_weight={cw}")
    ax[1].set(xscale="log", xlabel="C", title="Logistic regression", ylabel="CV PR-AUC")
    ax[1].legend(fontsize=8)

    rf = log[log["model"] == "random_forest"]
    sc = ax[2].scatter(rf["params"].apply(lambda q: q["max_depth"]), rf["cv_pr_auc_mean"],
                       c=rf["params"].apply(lambda q: q["min_samples_leaf"]), cmap="viridis")
    fig.colorbar(sc, ax=ax[2], label="min_samples_leaf")
    ax[2].set(xlabel="max_depth", title="Random forest", ylabel="CV PR-AUC")

    xg = log[log["model"] == "xgboost"]
    sc = ax[3].scatter(xg["params"].apply(lambda q: q["learning_rate"]), xg["cv_pr_auc_mean"],
                       c=xg["params"].apply(lambda q: q["max_depth"]), cmap="plasma")
    fig.colorbar(sc, ax=ax[3], label="max_depth")
    ax[3].set(xscale="log", xlabel="learning_rate", title="XGBoost", ylabel="CV PR-AUC")
    save(fig, "optuna_landscape.png")

    # Redraw each model's Optuna history from the log (same data as the training-time plot)
    for m in NAMES:
        vals = log.loc[log["model"] == m].sort_values("trial")["cv_pr_auc_mean"].values
        fig2, a2 = plt.subplots(figsize=(7, 4))
        a2.plot(vals, "o", alpha=0.5, label="trial")
        a2.plot(np.maximum.accumulate(vals), "-", label="best so far")
        a2.set(xlabel="trial", ylabel="PR-AUC (5-fold CV)", title=f"Optuna: {m}")
        a2.legend()
        save(fig2, f"optuna_{m}_history.png")

    for key, m in [("Log", "logreg"), ("Rf", "random_forest"), ("Xgb", "xgboost")]:
        v = log.loc[log["model"] == m, "cv_pr_auc_mean"]
        S[f"TrialMin{key}"], S[f"TrialMed{key}"], S[f"TrialMax{key}"] = f3(v.min()), f3(v.median()), f3(v.max())
        S[f"Within{key}"] = str(int((v >= v.max() - 0.005).sum()))
        S[f"FoldStd{key}"] = f3(float(log.loc[log["model"] == m].sort_values("cv_pr_auc_mean").iloc[-1]["cv_pr_auc_std"]))
    best_cw = lambda df, key, val: f3(df[df["params"].apply(lambda q: q[key] == val)]["cv_pr_auc_mean"].max())
    S["LogBestNone"], S["LogBestBal"] = best_cw(lr, "class_weight", None), best_cw(lr, "class_weight", "balanced")
    S["XgbBestW"], S["XgbBestNoW"] = best_cw(xg, "reweight", True), best_cw(xg, "reweight", False)
    S["RfBestNone"], S["RfBestBal"] = best_cw(rf, "class_weight", None), best_cw(rf, "class_weight", "balanced")
    S["TimeLog"] = f"{lr['duration_s'].median():.0f}"
    S["TimeRf"] = f"{rf['duration_s'].median():.0f}"
    S["TimeXgb"] = f"{xg['duration_s'].median():.0f}"


def shap_dependence(cfg, art, X_te):
    pipe = art["pipeline"]
    pre, model = pipe[:-1], pipe[-1]
    Xs = X_te.sample(n=1000, random_state=cfg["seed"])
    Z = pd.DataFrame(pre.transform(Xs), columns=pre[-1].get_feature_names_out(), index=Xs.index)
    sv = shap.TreeExplainer(model)(Z).values
    feats = ["age", "education-num", "hours-per-week", "log_capital_net"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, f in zip(axes, feats):
        j = list(Z.columns).index(f)
        sc = ax.scatter(Z[f], sv[:, j], c=Z["is_married"], cmap="coolwarm", s=8, alpha=0.6)
        ax.axhline(0, c="grey", lw=0.8)
        ax.set(xlabel=f, ylabel="SHAP value (log-odds)", title=f"Dependence: {f}")
    fig.colorbar(sc, ax=axes[-1], label="is_married")
    save(fig, "shap_dependence.png")

    def crossing(f, step=1):
        j = list(Z.columns).index(f)
        g = pd.Series(sv[:, j]).groupby((Z[f].values // step) * step).mean()
        pos = [v for v, m in g.items() if m > 0]
        return pos[0] if pos else float("nan")

    S["AgeCross"] = f"{crossing('age', 5):.0f}"
    S["EduCross"] = f"{crossing('education-num'):.0f}"
    S["HoursCross"] = f"{crossing('hours-per-week', 5):.0f}"
    j = list(Z.columns).index("is_married")
    S["MarriedShapOn"] = f"{sv[Z['is_married'] == 1, j].mean():+.2f}"
    S["MarriedShapOff"] = f"{sv[Z['is_married'] == 0, j].mean():+.2f}"


def group_figures(art, X_te, y_te):
    pipe, thr = art["pipeline"], art["threshold"]
    proba = pipe.predict_proba(X_te)[:, 1]
    sex = X_te["sex"].values
    yv = y_te.values

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for g, c in [("Female", "#4C72B0"), ("Male", "#DD8452")]:
        for cls, ls in [(0, ":"), (1, "-")]:
            m = (sex == g) & (yv == cls)
            ax[0].hist(proba[m], bins=40, histtype="step", density=True, color=c, ls=ls,
                       label=f"{g}, true {'>50K' if cls else '<=50K'}")
    ax[0].axvline(thr, c="k", ls="--")
    ax[0].set(title="Predicted probability by sex and true class (test)", xlabel="p(>50K)", ylabel="density")
    ax[0].legend(fontsize=7)

    ts = np.linspace(0.02, 0.98, 97)
    for g, c in [("Female", "#4C72B0"), ("Male", "#DD8452")]:
        m = sex == g
        sel = [(proba[m] >= t).mean() for t in ts]
        tpr = [(proba[m & (yv == 1)] >= t).mean() for t in ts]
        fpr = [(proba[m & (yv == 0)] >= t).mean() for t in ts]
        ax[1].plot(ts, tpr, c=c, label=f"TPR {g}")
        ax[1].plot(ts, fpr, c=c, ls="--", label=f"FPR {g}")
        ax[1].plot(ts, sel, c=c, ls=":", label=f"selection {g}")
    ax[1].axvline(thr, c="k", ls="--", lw=0.8)
    ax[1].set(title="Group rates vs one shared threshold (test)", xlabel="threshold", ylabel="rate")
    ax[1].legend(fontsize=7, ncol=2)

    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    for g, c in [("Female", "#4C72B0"), ("Male", "#DD8452")]:
        m = sex == g
        from sklearn.calibration import calibration_curve
        fp, mp = calibration_curve(yv[m], proba[m], n_bins=10, strategy="quantile")
        ax[2].plot(mp, fp, "o-", c=c, label=f"{g} (ECE={ece(yv[m], proba[m]):.3f})")
        key = "F" if g == "Female" else "M"
        S[f"Auc{key}"] = f3(roc_auc_score(yv[m], proba[m]))
        S[f"Ap{key}"] = f3(average_precision_score(yv[m], proba[m]))
        S[f"Ece{key}"] = f3(ece(yv[m], proba[m]))
        S[f"MeanP{key}"] = f3(proba[m].mean())
        S[f"Obs{key}"] = f3(yv[m].mean())
        S[f"MeanPPos{key}"] = f3(proba[m & (yv == 1)].mean())
        S[f"MeanPNeg{key}"] = f3(proba[m & (yv == 0)].mean())
    ax[2].set(title="Reliability by sex (test, raw model)", xlabel="mean predicted probability",
              ylabel="observed share >50K")
    ax[2].legend(fontsize=8)
    save(fig, "group_diagnostics.png")


def main():
    cfg = load_config()
    X_tr, y_tr = load_split(cfg, "train")
    X_va, y_va = load_split(cfg, "val")
    X_te, y_te = load_split(cfg, "test")
    models = {m: joblib.load(p(f"models/{m}.joblib")) for m in cfg["models"]}
    eda_extra(X_tr, y_tr)
    model_figures(models, X_va, y_va, X_te, y_te)
    optuna_landscape(cfg)
    shap_dependence(cfg, models["xgboost"], X_te)
    group_figures(models["xgboost"], X_te, y_te)
    with open(p("runs/extra_stats.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=2)
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
