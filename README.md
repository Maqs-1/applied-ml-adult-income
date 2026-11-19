# Applied ML — Adult / Census Income (Groupe 10)

Projet final du cours *Applied ML: Foundations, Engineering & Evaluation* (MSA-DATI07-05).

- **Dataset : Option 1 — Adult / Census Income** (`fetch_openml(name="adult", version=2)`), 48 842 personnes, 14 attributs.
- **Tâche :** prédire si le revenu annuel dépasse 50 K$ (classification binaire déséquilibrée, ~24 % de positifs).
- **Membres :** Max GUIRIEC, Amaniel ASRAT, Zackary AICH.
- **Rapport :** [`report/main.pdf`](report/main.pdf) (source LaTeX : `report/main.tex`).

## Résultats en bref

Modèle final : **XGBoost** (choisi sur la PR-AUC de validation), évalué **une seule fois** sur le test (9 758 lignes).

| Modèle | PR-AUC | ROC-AUC | F1 | Brier |
|---|---|---|---|---|
| Baseline (prévalence, validation) | 0,239 | 0,500 | 0,000 | 0,182 |
| Régression logistique | 0,740 | 0,902 | 0,681 | 0,105 |
| Random forest | 0,826 | 0,926 | 0,729 | 0,113 |
| **XGBoost** | **0,845** | **0,936** | **0,741** | 0,105 |

- **Calibration :** le modèle brut surestime les probabilités (ECE 0,106). Une recalibration isotonique, apprise sur la validation, ramène l'ECE à 0,011 et le Brier à 0,084.
- **Équité (sexe) :** l'écart d'*equalized odds* passe de 0,106 à 0,009 avec `ThresholdOptimizer`, au prix du F1 (0,741 → 0,681). La parité démographique n'est pas atteinte (0,231 → 0,124), ce qui est attendu quand les taux de base diffèrent.

Tous les chiffres sont produits par le code et stockés dans `runs/*.json` ; le rapport les relit automatiquement (`report/gen_numbers.py`).

## Installation (Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Les versions sont figées dans `requirements.txt`. Les dossiers `data/` et `models/*.joblib` ne sont pas versionnés : ils sont **recréés par l'exécution** des étapes ci-dessous.

## Exécution (dans l'ordre)

```bash
python -m src.data          # 1. EDA + split stratifié 60/20/20 (train/val/test)
python -m src.train         # 3-7. Optuna (CV 5 plis) pour logreg / RF / XGBoost + journal des essais
python -m src.evaluate      # 4. sélection sur la validation, puis évaluation unique sur le test
python -m src.explain       # 8. SHAP global et local
python -m src.calibration   # 9. courbe de fiabilité, Brier, ECE + recalibration isotonique
python -m src.fairness      # 10. audit d'équité (sexe, race) + ThresholdOptimizer
python -m src.fairness_ci   # 10. intervalles bootstrap 95 % de l'écart d'equalized odds
uvicorn src.api:app --port 8000   # endpoint de prédiction (nécessite models/xgboost.joblib)
```

Ou presque tout d'un coup : `python run_all.py` (enchaîne `data` → `extra_figures` ; lancer ensuite `python -m src.fairness_ci`).
La phase d'optimisation (120 essais Optuna au total) est la plus longue : comptez de l'ordre de 30 minutes à 1 heure selon la machine.

**Tests :**

```bash
python -m pytest -q         # métriques, calibration (ECE), écarts d'équité sur un cas jouet
```

**Données :** `src/data.py` utilise `fetch_openml` comme demandé. Si OpenML est indisponible (erreurs HTTP 503 constatées), le chargeur bascule sur les fichiers UCI d'origine, qui contiennent exactement les mêmes 48 842 lignes.

Tous les paramètres (graine 42, tailles des splits, nombre d'essais Optuna, attribut sensible…) sont dans `config.yaml`.

## Structure

| Fichier | Rôle | Exigence |
|---|---|---|
| `src/data.py` | chargement, EDA, split honnête (doublons retirés, stratifié cible × sexe) | 1 |
| `src/features.py` | `FeatureEngineer` (5 variables) + préprocessing, **un seul `Pipeline`** | 2, 5 |
| `src/train.py` | 3 modèles, repondération des classes, Optuna en CV, journal | 3, 4, 6, 7 |
| `src/metrics.py` | métriques communes (PR-AUC, ROC-AUC, F1, Brier, ECE…) | 3, 4, 9 |
| `src/evaluate.py` | évaluation finale sur le test | 4 |
| `src/explain.py` | SHAP global (beeswarm, barres) et local (4 cas) | 8 |
| `src/calibration.py` | courbe de fiabilité, Brier, ECE, recalibration | 9 |
| `src/fairness.py` | audit d'équité + mitigation `ThresholdOptimizer` | 10 |
| `src/fairness_ci.py` | intervalles de confiance bootstrap de l'écart d'équité | 10 |
| `src/extra_figures.py` | figures et statistiques d'analyse supplémentaires du rapport (ne change aucun résultat) | 1, 4, 8, 9, 10 |
| `src/api.py` | endpoint FastAPI `/health` et `/predict` | — |
| `tests/` | tests unitaires (métriques, calibration, équité) | — |
| `runs/trials_log.csv` | **journal de tous les essais Optuna** (paramètres, métriques CV, artefact associé) | 7 |
| `runs/*.json` | résultats chiffrés de chaque étape | — |
| `figures/` | toutes les figures du rapport | — |
| `report/` | rapport LaTeX et PDF, générateur des chiffres | — |
| `oral/` | préparation de la soutenance | — |

## Garanties méthodologiques

- Le **test** n'est lu que dans `evaluate.py`, `explain.py`, `calibration.py`, `fairness.py` et `fairness_ci.py`, après que tous les choix (hyperparamètres, seuil, modèle final, recalibration, mitigation) ont été faits sur train/validation.
- Tout le préprocessing est dans le `Pipeline`, donc ré-appris dans chaque pli de CV : aucune fuite.
- Optuna optimise la PR-AUC moyenne d'une CV stratifiée à 5 plis **sur le train uniquement** ; le seuil de décision est choisi sur la validation.
- `sex` et `race` sont exclus des entrées du modèle, mais conservés pour l'audit et la mitigation. Cela ne rend pas le modèle aveugle au sexe : `marital-status` et `relationship` en sont des proxies, ce que le rapport mesure.

## Exemple d'appel à l'API

Après avoir lancé `uvicorn src.api:app --port 8000` (les modèles doivent exister dans `models/`) :

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"age\":45,\"workclass\":\"Private\",\"education-num\":13,\"marital-status\":\"Married-civ-spouse\",\"occupation\":\"Exec-managerial\",\"relationship\":\"Husband\",\"race\":\"White\",\"sex\":\"Male\",\"capital-gain\":0,\"capital-loss\":0,\"hours-per-week\":50,\"native-country\":\"United-States\"}"
```

La réponse contient la probabilité de dépasser 50 K$, le seuil appris et la décision.

## Usage d'assistants IA

Claude Code (Anthropic) a servi d'assistant d'écriture et de débogage du code, de rédaction du rapport et de traduction en anglais. Les chiffres du rapport proviennent de l'exécution du code. Chaque membre répond de l'ensemble du projet. Détails dans la section « Use of AI » du rapport.

## Limites connues

Un seul split (pas de bootstrap sur les métriques de modèle), Optuna n'optimise que la PR-AUC, SHAP, calibration et équité sont rapportés sur le même test, et les données datent de 1994. Voir la section 9 du rapport.
