"""Exigences 3, 4, 6, 7 — Déséquilibre, 3 modèles, tuning Optuna en CV, journal des essais.

Usage : python -m src.train [--n-trials 40] [--models logreg random_forest xgboost]

Protocole (test jamais touché ici) :
  1. Optuna, sur le TRAIN seul, maximise la PR-AUC moyenne d'une CV stratifiée 5 plis.
     Le score rapporté est donc une moyenne hors-pli, pas un score sur données vues.
  2. Chaque essai est journalisé dans runs/trials_log.csv (params, métriques CV, artefact).
  3. Le meilleur jeu de paramètres est réentraîné sur tout le train ; le seuil de décision
     est choisi sur la VALIDATION (max F1) ; le modèle est sauvegardé.
  4. Un baseline (DummyClassifier) sert de point zéro de la progression.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.metrics import best_f1_threshold, compute_metrics
from src.features import build_pipeline
from src.utils import load_config, load_split, p, set_seed

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1",
           "balanced_acc": "balanced_accuracy", "brier": "neg_brier_score"}
LOG_FIELDS = ["model", "trial", "state", "params", "cv_pr_auc_mean", "cv_pr_auc_std",
              "cv_roc_auc_mean", "cv_f1_mean", "cv_balanced_acc_mean", "cv_brier_mean",
              "duration_s", "artifact", "is_best"]


def make_model(name: str, trial: optuna.Trial | None, params: dict | None, seed: int, pos_weight: float):
    """Retourne (estimateur, scale) depuis un essai Optuna ou des params figés.

    Remède au déséquilibre = repondération des classes, traitée comme hyperparamètre
    (class_weight None/balanced ; scale_pos_weight 1 ou n_neg/n_pos pour XGBoost),
    puis seuil de décision ajusté sur la validation."""
    P = params or {}

    def s(kind, key, *a, **kw):
        if trial is None:
            return P[key]
        return getattr(trial, f"suggest_{kind}")(key, *a, **kw)

    if name == "logreg":
        # Justification des plages (Régression Logistique) :
        # C : 1e-3 à 1e2, log scale, pour couvrir un large éventail de régularisation
        # class_weight : None ou balanced, pour gérer le déséquilibre des classes
        # liée au One-Hot Encoding, jusqu'à un modèle peu régularisé.
        m = LogisticRegression(C=s("float", "C", 1e-3, 1e2, log=True),
                               class_weight=s("categorical", "class_weight", [None, "balanced"]),
                               max_iter=3000, solver="lbfgs")
        return m, True
    if name == "random_forest":
        # Justification des plages (Forêt Aléatoire) :
        # n_estimators : 100 à 400, step=50, pour couvrir un large éventail de complexité
        # max_depth : 4 à 24, pour gérer la profondeur de l'arbre
        # min_samples_leaf : 1 à 30, pour contrôler la taille des feuilles
        # max_features : ["sqrt", "log2", 0.3, 0.5], pour gérer le nombre de caractéristiques testées à chaque division
        # class_weight : [None, "balanced", "balanced_subsample"], pour gérer le déséquilibre des classes
        m = RandomForestClassifier(
            n_estimators=s("int", "n_estimators", 100, 400, step=50),
            max_depth=s("int", "max_depth", 4, 24),
            min_samples_leaf=s("int", "min_samples_leaf", 1, 30),
            max_features=s("categorical", "max_features", ["sqrt", "log2", 0.3, 0.5]),
            class_weight=s("categorical", "class_weight", [None, "balanced", "balanced_subsample"]),
            n_jobs=-1, random_state=seed)
        return m, False
    if name == "xgboost":
        # Justification des plages (XGBoost) :
        # n_estimators : 100 à 500, step=50, pour couvrir un large éventail de complexité
        # learning_rate : 0.01 à 0.3, log scale, pour gérer la vitesse d'apprentissage
        # max_depth : 2 à 8, pour gérer la profondeur de l'arbre
        # min_child_weight : 1 à 20, log scale, pour contrôler la taille minimale des nœuds
        use_w = s("categorical", "reweight", [False, True])
        m = XGBClassifier(
            n_estimators=s("int", "n_estimators", 100, 500, step=50),
            learning_rate=s("float", "learning_rate", 0.01, 0.3, log=True),
            max_depth=s("int", "max_depth", 2, 8),
            min_child_weight=s("float", "min_child_weight", 1, 20, log=True),
            subsample=s("float", "subsample", 0.5, 1.0),
            colsample_bytree=s("float", "colsample_bytree", 0.4, 1.0),
            reg_lambda=s("float", "reg_lambda", 1e-3, 10, log=True),
            scale_pos_weight=pos_weight if use_w else 1.0,
            tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed)
        return m, False
    raise ValueError(name)


def tune(name: str, cfg: dict, X, y, writer, fh) -> optuna.Study:
    seed = cfg["seed"]
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    artifact = f"{cfg['paths']['figures_dir']}/optuna_{name}_history.png"

    def objective(trial):
        t0 = time.time()
        model, scale = make_model(name, trial, None, seed, pos_weight)
        pipe = build_pipeline(model, cfg, scale)
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        m = {k: float(np.mean(res[f"test_{k}"])) for k in SCORING}
        m["brier"] = -m["brier"]
        trial.set_user_attr("metrics", m)
        writer.writerow({
            "model": name, "trial": trial.number, "state": "COMPLETE",
            "params": json.dumps(trial.params),
            "cv_pr_auc_mean": round(m["pr_auc"], 5),
            "cv_pr_auc_std": round(float(np.std(res["test_pr_auc"])), 5),
            "cv_roc_auc_mean": round(m["roc_auc"], 5), "cv_f1_mean": round(m["f1"], 5),
            "cv_balanced_acc_mean": round(m["balanced_acc"], 5), "cv_brier_mean": round(m["brier"], 5),
            "duration_s": round(time.time() - t0, 1), "artifact": artifact, "is_best": "",
        })
        fh.flush()
        return m["pr_auc"]

    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], timeout=cfg["optuna"]["timeout_s"])

    vals = [t.value for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vals, "o", alpha=0.5, label="essai")
    ax.plot(np.maximum.accumulate(vals), "-", label="meilleur cumulé")
    ax.set(xlabel="essai", ylabel="PR-AUC (CV 5 plis)", title=f"Optuna — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p(artifact), dpi=110)
    plt.close(fig)
    return study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    cfg = load_config()
    if args.n_trials:
        cfg["optuna"]["n_trials"] = args.n_trials
    models = args.models or cfg["models"]
    seed = cfg["seed"]
    set_seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr, y_tr = load_split(cfg, "train")
    X_va, y_va = load_split(cfg, "val")
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    results = {}
    # Baseline : prédit toujours la classe majoritaire (probabilité = prévalence)
    dummy = build_pipeline(DummyClassifier(strategy="prior"), cfg, scale=False).fit(X_tr, y_tr)
    results["baseline_dummy"] = {"val": compute_metrics(y_va, dummy.predict_proba(X_va)[:, 1], 0.5)}

    log_path = p(cfg["paths"]["trials_log"])
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for name in models:
            print(f"== Tuning {name} ({cfg['optuna']['n_trials']} essais)")
            study = tune(name, cfg, X_tr, y_tr, writer, fh)
            best = study.best_trial
            model, scale = make_model(name, None, best.params, seed, pos_weight)
            pipe = build_pipeline(model, cfg, scale).fit(X_tr, y_tr)
            proba_va = pipe.predict_proba(X_va)[:, 1]
            thr = best_f1_threshold(y_va, proba_va)
            model_path = f"{cfg['paths']['models_dir']}/{name}.joblib"
            joblib.dump({"pipeline": pipe, "threshold": thr, "params": best.params}, p(model_path))
            writer.writerow({"model": name, "trial": best.number, "state": "REFIT_BEST",
                             "params": json.dumps(best.params),
                             **{f"cv_{k}_mean": round(v, 5) for k, v in best.user_attrs["metrics"].items()},
                             "artifact": model_path, "is_best": "yes"})
            fh.flush()
            results[name] = {
                "best_params": best.params, "n_trials": len(study.trials),
                "cv": best.user_attrs["metrics"],
                "val_threshold": thr,
                "val_at_0.5": compute_metrics(y_va, proba_va, 0.5),
                "val_at_thr": compute_metrics(y_va, proba_va, thr),
            }
            print(json.dumps(results[name], indent=2))

    with open(p("runs/train_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
