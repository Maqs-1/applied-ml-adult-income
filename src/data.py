"""Exigence 1 — Chargement, EDA et split honnête train/val/test.

Usage : python -m src.data
Sorties : data/splits/{train,val,test}.parquet, runs/eda_summary.json, figures/eda_*.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.utils import load_config, p, set_seed


UCI_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]


def _from_uci() -> pd.DataFrame:
    """Repli si OpenML est indisponible : même jeu (adult.data + adult.test = 48 842 lignes,
    identique à OpenML adult v2), depuis la source UCI d'origine."""
    import io
    import urllib.request
    import zipfile

    raw = urllib.request.urlopen(UCI_URL, timeout=60).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    parts = []
    for name, skip in [("adult.data", 0), ("adult.test", 1)]:
        d = pd.read_csv(zf.open(name), header=None, names=COLS + ["income"], skiprows=skip,
                        skipinitialspace=True, na_values="?")
        parts.append(d.dropna(how="all"))
    df = pd.concat(parts, ignore_index=True)
    df["target"] = (df.pop("income").str.rstrip(".") == ">50K").astype(int)
    return df


def load_raw(cfg: dict) -> pd.DataFrame:
    path = p(cfg["data"]["raw_path"])
    if path.exists():
        return pd.read_parquet(path)
    try:
        ds = fetch_openml(name=cfg["data"]["openml_name"], version=cfg["data"]["openml_version"], as_frame=True)
        df = ds.data.copy()
        df["target"] = (ds.target.astype(str) == cfg["data"]["positive_label"]).astype(int)
    except Exception as e:  # OpenML renvoie parfois 503
        print(f"OpenML indisponible ({e}); repli sur la source UCI.")
        df = _from_uci()
    # Les catégories pandas posent problème en parquet + sklearn : on repasse en object
    for c in df.select_dtypes("category").columns:
        df[c] = df[c].astype(object)
    df.to_parquet(path)
    return df


def eda(df: pd.DataFrame, cfg: dict) -> dict:
    sens = cfg["features"]["sensitive"]
    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1] - 1),
        "n_exact_duplicates": int(df.duplicated().sum()),
        "positive_rate": round(float(df["target"].mean()), 4),
        "missing_by_col": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "positive_rate_by_sex": df.groupby(sens)["target"].mean().round(4).to_dict(),
        "share_by_sex": df[sens].value_counts(normalize=True).round(4).to_dict(),
        "positive_rate_by_race": df.groupby("race")["target"].mean().round(4).to_dict(),
        "capital_gain_zero_share": round(float((df["capital-gain"] == 0).mean()), 4),
        "capital_loss_zero_share": round(float((df["capital-loss"] == 0).mean()), 4),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    df["target"].value_counts().sort_index().plot.bar(ax=axes[0, 0], color=["#4C72B0", "#DD8452"])
    axes[0, 0].set_title("Classe cible (0 = <=50K, 1 = >50K)")
    for ax, col in zip([axes[0, 1], axes[0, 2]], ["age", "hours-per-week"]):
        df.groupby("target")[col].plot.hist(ax=ax, bins=30, alpha=0.6, legend=True)
        ax.set_title(f"{col} par classe")
    df.groupby("education-num")["target"].mean().plot(ax=axes[1, 0], marker="o")
    axes[1, 0].set_title("Taux >50K vs education-num")
    df.groupby(sens)["target"].mean().plot.bar(ax=axes[1, 1], color="#55A868")
    axes[1, 1].set_title("Taux >50K par sexe")
    df.groupby("marital-status")["target"].mean().sort_values().plot.barh(ax=axes[1, 2], color="#C44E52")
    axes[1, 2].set_title("Taux >50K par statut marital")
    fig.tight_layout()
    fig.savefig(p("figures/eda_overview.png"), dpi=120)
    plt.close(fig)
    return summary


def split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Split stratifié sur (cible x sexe) 60/20/20. Doublons exacts retirés AVANT le split
    pour éviter qu'une même ligne se retrouve à la fois en train et en test (fuite)."""
    seed = cfg["seed"]
    df = df.drop_duplicates().reset_index(drop=True)
    strat = df["target"].astype(str) + "_" + df[cfg["features"]["sensitive"]].astype(str)
    test_frac = cfg["data"]["test_size"]
    val_frac = cfg["data"]["val_size"] / (1 - test_frac)
    trval, test = train_test_split(df, test_size=test_frac, stratify=strat, random_state=seed)
    train, val = train_test_split(trval, test_size=val_frac, stratify=strat.loc[trval.index], random_state=seed)
    return {"train": train, "val": val, "test": test}


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    df = load_raw(cfg)
    summary = eda(df, cfg)
    splits = split(df, cfg)
    for name, d in splits.items():
        d.to_parquet(p(f"{cfg['data']['splits_dir']}/{name}.parquet"))
    summary["splits"] = {k: {"n": len(v), "positive_rate": round(float(v["target"].mean()), 4)} for k, v in splits.items()}
    with open(p("runs/eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
