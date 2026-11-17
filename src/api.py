"""Endpoint de prédiction (« served prediction endpoint » de la consigne).

Lancer : uvicorn src.api:app --port 8000
Tester : POST http://localhost:8000/predict avec un JSON (voir README).
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.utils import load_config, p

cfg = load_config()
FINAL = json.loads(p(cfg["paths"]["results"]).read_text(encoding="utf-8"))["final_model"]
ART = joblib.load(p(f"models/{FINAL}.joblib"))

app = FastAPI(title="Adult income — prédiction >50K", version="1.0")


class Person(BaseModel):
    age: int
    workclass: str | None = None
    fnlwgt: float = 0.0
    education: str | None = None
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str | None = None
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(0.0, alias="capital-gain")
    capital_loss: float = Field(0.0, alias="capital-loss")
    hours_per_week: float = Field(alias="hours-per-week")
    native_country: str | None = Field(None, alias="native-country")


@app.get("/health")
def health():
    return {"status": "ok", "model": FINAL}


@app.post("/predict")
def predict(person: Person):
    X = pd.DataFrame([person.model_dump(by_alias=True)])
    proba = float(ART["pipeline"].predict_proba(X)[0, 1])
    return {"model": FINAL, "proba_gt_50k": round(proba, 4), "threshold": round(ART["threshold"], 4),
            "prediction": ">50K" if proba >= ART["threshold"] else "<=50K"}
