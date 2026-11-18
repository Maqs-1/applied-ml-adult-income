"""Builds report/generated.tex (LaTeX macros + tables) from the runs/*.json files.

No number in the report is typed by hand: everything comes from running the code.
Usage: python report/gen_numbers.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"
NAMES = {"logreg": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost",
         "baseline_dummy": "Baseline (prior)"}
CASES = {"vrai_positif": "true positive", "vrai_negatif": "true negative",
         "faux_positif": "false positive", "faux_negatif": "false negative"}


def j(name):
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def macro(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def f3(x):
    return f"{x:.3f}"


def main():
    eda, tr, res = j("eda_summary.json"), j("train_results.json"), j("results.json")
    cal, fair = j("calibration.json"), j("fairness.json")
    shap_local = j("shap_local.json")
    out = []
    final = res["final_model"]
    out.append(macro("FinalModel", NAMES[final]))
    out.append(macro("NRows", f"{eda['n_rows']:,}".replace(",", "\\,")))
    out.append(macro("NDup", eda["n_exact_duplicates"]))
    out.append(macro("PosRate", f"{100 * eda['positive_rate']:.1f}"))
    out.append(macro("PosF", f"{100 * eda['positive_rate_by_sex']['Female']:.1f}"))
    out.append(macro("PosM", f"{100 * eda['positive_rate_by_sex']['Male']:.1f}"))
    out.append(macro("ShareF", f"{100 * eda['share_by_sex']['Female']:.1f}"))
    out.append(macro("CapZero", f"{100 * eda['capital_gain_zero_share']:.1f}"))
    for k in ("train", "val", "test"):
        out.append(macro(f"N{k.capitalize()}", f"{eda['splits'][k]['n']:,}".replace(",", "\\,")))
    miss = eda["missing_by_col"]
    out.append(macro("MissWork", miss.get("workclass", 0)))
    out.append(macro("MissOcc", miss.get("occupation", 0)))
    out.append(macro("MissCountry", miss.get("native-country", 0)))

    # Tableau CV (train) + validation
    rows = []
    for m in ["logreg", "random_forest", "xgboost"]:
        r = tr[m]
        rows.append(f"{NAMES[m]} & {r['n_trials']} & {f3(r['cv']['pr_auc'])} & {f3(r['cv']['roc_auc'])} & "
                    f"{f3(r['val_at_thr']['pr_auc'])} & {f3(r['val_at_thr']['f1'])} & {f3(r['val_threshold'])} \\\\")
    out.append("\\newcommand{\\TabCV}{" + "\n".join(rows) + "}\n")

    # Tableau test : mêmes métriques pour tous
    rows = []
    for m in ["logreg", "random_forest", "xgboost"]:
        t = res["test"][m]["at_val_threshold"]
        t5 = res["test"][m]["at_0.5"]
        bold = (lambda s: f"\\textbf{{{s}}}") if m == final else (lambda s: s)
        rows.append(bold(NAMES[m]) + " & " + " & ".join(f3(t[k]) for k in
                    ["pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_acc", "brier"])
                    + f" & {f3(t5['f1'])} \\\\")
    d = tr["baseline_dummy"]["val"]
    rows.insert(0, f"{NAMES['baseline_dummy']} (val.) & {f3(d['pr_auc'])} & {f3(d['roc_auc'])} & {f3(d['f1'])} & "
                   f"-- & {f3(d['recall'])} & {f3(d['balanced_acc'])} & {f3(d['brier'])} & -- \\\\")
    out.append("\\newcommand{\\TabTest}{" + "\n".join(rows) + "}\n")
    ft = res["test"][final]["at_val_threshold"]
    for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_acc", "brier", "ece"]:
        out.append(macro("Fin" + k.replace("_", "").replace("f1", "fone").capitalize(), f3(ft[k])))
    out.append(macro("FinThr", f3(ft["threshold"])))
    out.append(macro("FinFHalf", f3(res["test"][final]["at_0.5"]["f1"])))
    out.append(macro("FinRecallHalf", f3(res["test"][final]["at_0.5"]["recall"])))

    # Meilleurs hyperparamètres
    rows = []
    for m in ["logreg", "random_forest", "xgboost"]:
        bp = ", ".join(f"{esc(k)}={(f'{v:.3g}' if isinstance(v, float) else esc(v))}"
                       for k, v in tr[m]["best_params"].items())
        rows.append(f"{NAMES[m]} & \\footnotesize {bp} \\\\")
    out.append("\\newcommand{\\TabParams}{" + "\n".join(rows) + "}\n")

    # Calibration
    out.append(macro("CalBrier", f3(cal["raw"]["brier"])))
    out.append(macro("CalEce", f3(cal["raw"]["ece"])))
    out.append(macro("CalGap", f"{cal['raw']['mean_gap_pred_minus_obs']:+.3f}"))
    out.append(macro("CalMeanPred", f3(cal["raw"]["mean_pred"])))
    out.append(macro("CalObs", f3(cal["raw"]["observed_rate"])))
    out.append(macro("CalIsoBrier", f3(cal["isotonic_on_val"]["brier"])))
    out.append(macro("CalIsoEce", f3(cal["isotonic_on_val"]["ece"])))
    bins = list(zip(cal["raw"]["bins_mean_pred"], cal["raw"]["bins_frac_pos"]))
    out.append("\\newcommand{\\TabCalBins}{" + "\n".join(
        f"{i + 1} & {f3(a)} & {f3(b)} & {a - b:+.3f} \\\\" for i, (a, b) in enumerate(bins)) + "}\n")

    # Équité
    rows = []
    for tag, key in [("Before", "before"), ("After", "after_threshold_optimizer")]:
        a = fair[key]
        for g in ["Female", "Male"]:
            bg = a["by_group"][g]
            rows.append(f"{tag} & {g} & {f3(bg['selection_rate'])} & {f3(bg['tpr'])} & {f3(bg['fpr'])} & "
                        f"{f3(bg['precision'])} & {f3(bg['accuracy'])} \\\\")
        rows.append(f"{tag} & \\textit{{overall}} & {f3(a['overall']['selection_rate'])} & {f3(a['overall']['tpr'])} & "
                    f"{f3(a['overall']['fpr'])} & {f3(a['overall']['precision'])} & {f3(a['overall']['accuracy'])} \\\\ \\midrule")
    out.append("\\newcommand{\\TabFair}{" + "\n".join(rows) + "}\n")
    for tag, key in [("B", "before"), ("A", "after_threshold_optimizer")]:
        a = fair[key]
        out.append(macro(f"Dpd{tag}", f3(a["demographic_parity_diff"])))
        out.append(macro(f"Dpr{tag}", f3(a["demographic_parity_ratio"])))
        out.append(macro(f"Eod{tag}", f3(a["equalized_odds_diff"])))
        out.append(macro(f"Fone{tag}", f3(a["overall"]["f1"])))
        out.append(macro(f"Acc{tag}", f3(a["overall"]["accuracy"])))
    r = fair["secondary_race_before"]
    out.append(macro("RaceDpd", f3(r["demographic_parity_diff"])))
    out.append(macro("RaceEod", f3(r["equalized_odds_diff"])))
    rows = [f"{esc(g)} & {int(v['count'])} & {f3(v['selection_rate'])} & {f3(v['tpr'])} & {f3(v['fpr'])} \\\\"
            for g, v in r["by_group"].items()]
    out.append("\\newcommand{\\TabRace}{" + "\n".join(rows) + "}\n")

    # SHAP
    imp = list(csv.reader(open(RUNS / "shap_global.csv", encoding="utf-8")))[1:11]
    out.append("\\newcommand{\\TabShap}{" + "\n".join(
        f"{i + 1} & \\texttt{{{esc(n)}}} & {float(v):.3f} \\\\" for i, (n, v) in enumerate(imp)) + "}\n")
    rows = []
    for lab, c in shap_local["cases"].items():
        top = "; ".join(f"\\texttt{{{esc(k)}}} {v:+.2f}" for k, v in list(c["top_contributions"].items())[:4])
        rows.append(f"{CASES.get(lab, lab)} & {c['y_true']} & {f3(c['proba'])} & \\footnotesize {top} \\\\")
    out.append("\\newcommand{\\TabShapLocal}{" + "\n".join(rows) + "}\n")

    # Journal des essais (annexe)
    log = list(csv.DictReader(open(RUNS / "trials_log.csv", encoding="utf-8")))
    out.append(macro("NTrialsTotal", sum(1 for r in log if r["state"] == "COMPLETE")))
    by = {}
    for r in log:
        if r["state"] == "COMPLETE":
            by[(r["model"], int(r["trial"]))] = r
    n = max(t for _, t in by) + 1
    rows = []
    for t in range(n):
        cells = [str(t)]
        for m in ["logreg", "random_forest", "xgboost"]:
            r = by.get((m, t))
            cells += ([f"{float(r['cv_pr_auc_mean']):.4f}", f"{float(r['cv_pr_auc_std']):.4f}",
                       f"{float(r['cv_f1_mean']):.3f}", f"{float(r['duration_s']):.0f}"] if r else ["--"] * 4)
        rows.append(" & ".join(cells) + " \\\\")
    out.append("\\newcommand{\\TabLog}{" + "\n".join(rows) + "}\n")


    # Macros supplémentaires (prose du rapport)
    for tag, key in [("B", "before"), ("A", "after_threshold_optimizer")]:
        for g, gl in [("F", "Female"), ("M", "Male")]:
            bg = fair[key]["by_group"][gl]
            for mname, mk in [("Sel", "selection_rate"), ("Tpr", "tpr"), ("Fpr", "fpr"), ("Prec", "precision"), ("Acc", "accuracy")]:
                out.append(macro(f"{mname}{g}{tag}", f3(bg[mk])))
    for short, m in [("Log", "logreg"), ("Rf", "random_forest"), ("Xgb", "xgboost")]:
        t = res["test"][m]["at_val_threshold"]
        out.append(macro(f"TestPr{short}", f3(t["pr_auc"])))
        out.append(macro(f"TestRoc{short}", f3(t["roc_auc"])))
        out.append(macro(f"TestFone{short}", f3(t["f1"])))
        out.append(macro(f"CvPr{short}", f3(tr[m]["cv"]["pr_auc"])))
        out.append(macro(f"ValPr{short}", f3(tr[m]["val_at_thr"]["pr_auc"])))
        out.append(macro(f"Thr{short}", f3(tr[m]["val_threshold"])))
    b = tr["baseline_dummy"]["val"]
    out.append(macro("BasePr", f3(b["pr_auc"])))
    out.append(macro("BaseBrier", f3(b["brier"])))
    out.append(macro("BaseAcc", f3(b["accuracy"])))
    out.append(macro("FinAcc", f3(ft["accuracy"])))
    out.append(macro("CvStdXgb", f3(float(__import__("statistics").pstdev(
        [float(r["cv_pr_auc_mean"]) for r in csv.DictReader(open(RUNS / "trials_log.csv", encoding="utf-8"))
         if r["model"] == "xgboost" and r["state"] == "COMPLETE"])))))

    out.append(macro("EceLog", f3(res["test"]["logreg"]["at_val_threshold"]["ece"])))
    out.append(macro("BrierLog", f3(res["test"]["logreg"]["at_val_threshold"]["brier"])))
    out.append(macro("RealGap", f3(eda["positive_rate_by_sex"]["Male"] - eda["positive_rate_by_sex"]["Female"])))
    import joblib
    _pipe = joblib.load(ROOT / "models" / f"{final}.joblib")["pipeline"]
    out.append(macro("NCols", len(_pipe[:-1][-1].get_feature_names_out())))

    # Extra statistics (src/extra_figures.py), prefixed with X
    for k, v in j("extra_stats.json").items():
        out.append(macro("X" + k, v))

    (ROOT / "report" / "generated.tex").write_text("".join(out), encoding="utf-8")
    print("report/generated.tex written")


if __name__ == "__main__":
    main()
