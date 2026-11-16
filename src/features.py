"""Exigences 2 et 5 — Feature engineering et Pipeline sklearn unique.

Toutes les transformations vivent dans le Pipeline : elles sont apprises sur les plis
d'entraînement uniquement (pas de fuite en validation croisée ni vers le test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MARRIED = {"Married-civ-spouse", "Married-AF-spouse"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Ajoute des variables métier et retire les colonnes exclues.

    Transformation ligne à ligne, sans statistique apprise : aucune fuite possible.
    - log_capital_net : log signé de (capital-gain - capital-loss). Les gains en capital
      sont nuls pour ~92 % des gens et très asymétriques ; le log écrase les extrêmes.
    - has_capital : indicateur de revenus du capital (proxy de patrimoine).
    - is_married : marié(e) vivant avec conjoint — regroupe 7 modalités en un signal fort
      (foyer à deux revenus déclarés).
    - overtime : plus de 40 h/semaine (temps plein dépassé).
    - age_x_edu : interaction âge x années d'études (expérience valorisée par le diplôme),
      utile surtout au modèle linéaire qui ne crée pas d'interactions seul.
    """

    def __init__(self, drop: list[str] | None = None):
        self.drop = drop

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        net = X["capital-gain"].astype(float) - X["capital-loss"].astype(float)
        X["log_capital_net"] = np.sign(net) * np.log1p(np.abs(net))
        X["has_capital"] = ((X["capital-gain"] > 0) | (X["capital-loss"] > 0)).astype(int)
        X["is_married"] = X["marital-status"].isin(MARRIED).astype(int)
        X["overtime"] = (X["hours-per-week"] > 40).astype(int)
        X["age_x_edu"] = X["age"].astype(float) * X["education-num"].astype(float)
        X = X.drop(columns=["capital-gain", "capital-loss"])
        X = X.drop(columns=[c for c in (self.drop or []) if c in X.columns])
        return X


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Numériques : imputation médiane (+ standardisation pour le linéaire).
    Catégorielles : NaN -> modalité 'Missing' (le manque est informatif : ex. occupation
    inconnue = souvent hors emploi), puis one-hot ; modalités rares regroupées."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), make_column_selector(dtype_include=np.number)),
            ("cat", cat, make_column_selector(dtype_include=object)),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model, cfg: dict, scale: bool) -> Pipeline:
    drop = list(cfg["features"]["drop"]) + list(cfg["features"]["exclude_from_model"])
    return Pipeline([
        ("features", FeatureEngineer(drop=drop)),
        ("preprocess", build_preprocessor(scale=scale)),
        ("model", model),
    ])
