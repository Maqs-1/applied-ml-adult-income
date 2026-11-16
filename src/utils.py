"""Utilitaires partagés : config, graine, chargement des splits."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def p(rel: str) -> Path:
    """Chemin absolu à partir de la racine du projet (crée le dossier parent)."""
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def load_split(cfg: dict, name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Charge un split ('train', 'val', 'test'). Cible binaire : 1 = >50K."""
    df = pd.read_parquet(p(f"{cfg['data']['splits_dir']}/{name}.parquet"))
    y = df.pop("target").astype(int)
    return df, y
