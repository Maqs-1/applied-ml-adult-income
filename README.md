# Applied ML — Adult / Census Income (Groupe 10)

Projet final du cours *Applied ML: Foundations, Engineering & Evaluation* (MSA-DATI07-05).
**Dataset : Option 1 — Adult / Census Income** (`fetch_openml(name="adult", version=2)`).
Tâche : prédire si le revenu annuel dépasse 50 K$ (classification binaire déséquilibrée, ~24 % de positifs).

Membres : Max GUIRIEC, Amaniel ASRAT, Zackary AICH.

## Installation (Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution (dans l'ordre)

```bash
python -m src.data          # 1. EDA + split stratifié 60/20/20 (train/val/test)
python -m src.train         # 3-7. Optuna (CV 5 plis) pour logreg / RF / XGBoost + journal des essais
python -m src.evaluate      # 4. sélection sur la validation, puis évaluation unique sur le test
python -m src.explain       # 8. SHAP global et local
python -m src.calibration   # 9. courbe de fiabilité, Brier, ECE
python -m src.fairness      # 10. audit d'équité (sexe, race) + ThresholdOptimizer
uvicorn src.api:app --port 8000   # endpoint de prédiction
```

Ou tout d'un coup : `python run_all.py`.

Tous les paramètres (graine 42, tailles des splits, nombre d'essais Optuna, attribut sensible…) sont dans `config.yaml`.

## Structure

| Fichier | Rôle | Exigence |
|---|---|---|
| `src/data.py` | chargement, EDA, split honnête | 1 |
| `src/features.py` | `FeatureEngineer` + préprocessing, **un seul `Pipeline`** | 2, 5 |
| `src/train.py` | 3 modèles, repondération des classes, Optuna en CV, journal | 3, 4, 6, 7 |
| `src/metrics.py` | métriques communes (PR-AUC, ROC-AUC, F1, Brier, ECE…) | 3, 4, 9 |
| `src/evaluate.py` | évaluation finale sur le test | 4 |
| `src/explain.py` | SHAP | 8 |
| `src/calibration.py` | calibration | 9 |
| `src/fairness.py` | équité + mitigation | 10 |
| `src/extra_figures.py` | figures et statistiques d'analyse supplémentaires du rapport (ne change aucun résultat) | 1, 4, 8, 9, 10 |
| `src/api.py` | endpoint FastAPI `/predict` | — |
| `runs/trials_log.csv` | **journal de tous les essais Optuna** (params, métriques CV, artefact) | 7 |
| `runs/*.json` | résultats chiffrés de chaque étape | — |
| `figures/` | toutes les figures du rapport | — |
| `report/` | rapport LaTeX | — |

## Garanties méthodologiques

- Le **test** n'est lu que dans `evaluate.py`, `explain.py`, `calibration.py` et `fairness.py`, après que tous les choix (hyperparamètres, seuil, modèle final, recalibration, mitigation) ont été faits sur train/validation.
- Tout le préprocessing est dans le `Pipeline`, donc ré-appris dans chaque pli de CV : aucune fuite.
- `sex` et `race` sont exclus des entrées du modèle, mais conservés pour l'audit et la mitigation.

## Exemple d'appel à l'API

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"age\":45,\"workclass\":\"Private\",\"education-num\":13,\"marital-status\":\"Married-civ-spouse\",\"occupation\":\"Exec-managerial\",\"relationship\":\"Husband\",\"race\":\"White\",\"sex\":\"Male\",\"capital-gain\":0,\"capital-loss\":0,\"hours-per-week\":50,\"native-country\":\"United-States\"}"
```
