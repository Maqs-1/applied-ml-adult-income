# Préparation de l'oral : Adult / Census Income

**Quand :** vendredi 20/11/2026, **16h15**, salle **EXT+** (d'après « ORAL DEFENSE TIMETABLE UPDATED.pdf »). Être devant la salle **dès 16h05**.
**Format :** 7 min de présentation + 3 min de questions. Les trois membres parlent ; chacun doit pouvoir répondre sur **tout** le projet.

## 1. Plan des 7 minutes (8 diapositives, environ 50 s chacune)

| Temps | Qui | Diapo | Contenu | Chiffres à citer |
|---|---|---|---|---|
| 0:00-0:50 | **Max** | 1. Problème | Prédire revenu > 50 k\$ (Adult 1994), 48 842 lignes, 23,9 % de positifs. Pourquoi c'est déséquilibré et pourquoi PR-AUC. | baseline accuracy 0,761 mais rappel 0 ; baseline PR-AUC 0,239 |
| 0:50-1:40 | **Max** | 2. Données et split | EDA (manquants sur 3 colonnes, capital très asymétrique), doublons retirés, split 60/20/20 stratifié cible × sexe, test intouché. | 29 274 / 9 758 / 9 758 ; femmes 10,9 % vs hommes 30,4 % |
| 1:40-2:20 | **Max** | 3. Features et Pipeline | Un seul `Pipeline`, 5 variables créées, `sex` et `race` exclus du modèle, pas de fuite. | 76 colonnes |
| 2:20-3:10 | **Amaniel** | 4. Déséquilibre et modèles | Repondération traitée comme hyperparamètre + seuil sur la validation. Progression logistique, Random Forest, XGBoost. | seuil XGBoost 0,653 ; F1 0,722 (seuil 0,5) → 0,741 |
| 3:10-4:00 | **Amaniel** | 5. Optuna et résultats | 40 essais × 3 modèles, CV 5 plis sur le train, journal CSV. Tableau test. | PR-AUC test : 0,740 / 0,826 / **0,845** |
| 4:00-4:50 | **Zackary** | 6. SHAP | Variables dominantes plausibles ; le statut marital est un proxy possible du sexe. 4 cas locaux. | top : `is_married`, gains en capital, `age_x_edu` |
| 4:50-5:50 | **Zackary** | 7. Calibration | Sur-estime les positifs ; cause : repondération ; isotonique corrige. | proba moyenne 0,345 vs réel 0,239 ; ECE 0,106 → 0,011 |
| 5:50-7:00 | **Zackary** | 8. Équité et conclusion | Écarts avant/après, coût de la mitigation, limites. | EOD 0,106 → 0,009 ; F1 0,741 → 0,681 ; précision des femmes 0,699 → 0,335 |

Conseils : (1) répéter à voix haute au moins une fois avec un chronomètre ; (2) chacun connaît les chiffres des autres ; (3) message final : « un bon modèle ne suffit pas : on a mesuré sa calibration et son équité, et chaque correction a un prix ».

## 2. Les 20 questions probables (réponses courtes)

1. **Pourquoi PR-AUC et pas l'accuracy ?** Un modèle qui prédit toujours « ≤ 50K » a 0,761 d'accuracy et un rappel nul. La PR-AUC se concentre sur la classe rare, ne dépend pas d'un seuil, et son niveau de référence est la prévalence (0,239).
2. **Pourquoi pas SMOTE ?** Le déséquilibre est modéré (1 pour 3), les variables sont mixtes (binaires + continues) donc les points synthétiques sont peu naturels, et la repondération + le seuil suffisent. Un rééchantillonnage devrait de toute façon être dans le pipeline.
3. **Comment évitez-vous la fuite de données ?** Tout le prétraitement est dans un `Pipeline` donc appris pli par pli ; doublons retirés avant le split ; test utilisé seulement pour rapporter, aucun choix n'y est fait.
4. **Pourquoi stratifier sur cible × sexe ?** Pour garder la même prévalence et la même proportion de femmes/hommes dans les trois ensembles, ce qui rend l'audit d'équité comparable.
5. **Pourquoi retirer `fnlwgt`, `education`, `sex`, `race` ?** `fnlwgt` est un poids d'échantillonnage, pas une caractéristique ; `education` est redondante avec `education-num` ; `sex` et `race` sont gardés à part pour l'audit. Mais ça ne supprime pas le biais (proxys).
6. **Quelles variables avez-vous construites ?** `log_capital_net`, `has_capital`, `is_married`, `overtime`, `age_x_edu` ; chacune est justifiée (queue lourde, patrimoine, foyer, temps de travail, expérience × diplôme).
7. **Votre score Optuna est-il optimiste ?** Il est calculé en CV sur des plis non vus par chaque essai, mais le meilleur de 40 est sélectionné dessus, donc légèrement optimiste. C'est pourquoi on compare ensuite sur la validation puis le test.
8. **Comment fonctionne Optuna ?** L'échantillonneur TPE modélise les zones de paramètres qui donnent de bons et de mauvais scores et propose les prochains essais dans les bonnes zones. Nous n'avons pas utilisé de pruning.
9. **Pourquoi XGBoost comme modèle final ?** Meilleure PR-AUC en CV (0,826), validation (0,816) et test (0,845). L'écart avec le Random Forest est petit et nous n'avons pas calculé d'intervalle de confiance : c'est une limite assumée.
10. **Pourquoi choisir le seuil sur la validation ?** Pour maximiser le F1 sans toucher au test. Pour XGBoost, 0,653 donne un F1 de 0,741 contre 0,722 à 0,5 (précision plus haute, rappel plus bas).
11. **Le test (0,845) est au-dessus du CV (0,826) : suspect ?** Non : le test n'a servi à aucun choix. L'écart est de l'ordre de la variabilité d'un échantillon de cette taille ; nous n'avons pas de bootstrap pour le chiffrer.
12. **Comment marche SHAP ?** Valeurs de Shapley : contribution moyenne de chaque variable à l'écart entre la prédiction et la prédiction moyenne, sur tous les ordres d'ajout. `TreeExplainer` est exact pour les arbres ; les contributions s'additionnent en log-odds.
13. **Les variables importantes ont-elles du sens ?** Oui : statut marital, gains en capital, études, âge, heures. Mais le statut marital est aussi un proxy possible du sexe (`relationship_Husband` est dans le top 10), hypothèse non mesurée.
14. **Pourquoi `is_married` et `Married-civ-spouse` sont tous deux en haut ?** Ce sont deux encodages quasi identiques ; SHAP répartit le crédit entre variables corrélées, donc leur importance se cumule.
15. **Le modèle est-il sur- ou sous-confiant ?** Il sur-estime la classe positive : probabilité moyenne 0,345 pour un taux réel de 0,239 ; ECE 0,106 ; Brier 0,105.
16. **Feriez-vous confiance à ses probabilités ?** Pas brutes : la repondération les décale vers le haut. Après calibration isotonique sur la validation, l'ECE tombe à 0,011 et le Brier à 0,084. Pour un simple classement ou avec le seuil appris, le modèle brut convient.
17. **Brier et ECE, c'est quoi ?** Brier = erreur quadratique moyenne entre probabilité et issue (0 ou 1). ECE = écart moyen, pondéré par bac, entre probabilité prédite et fréquence observée (10 bacs de largeur égale ; la courbe de fiabilité, elle, utilise des bacs par quantiles).
18. **Parité démographique vs equalized odds ?** Parité : même taux de prédictions positives entre groupes. Equalized odds : mêmes TPR et FPR. Les taux de base diffèrent (10,9 % vs 30,4 %), donc on a visé l'equalized odds.
19. **Quel est le coût de votre mitigation ?** L'écart d'equalized odds passe de 0,106 à 0,009 et celui de parité de 0,231 à 0,124, mais le F1 baisse de 0,741 à 0,681 et la précision des femmes s'effondre de 0,699 à 0,335 (FPR 0,037 → 0,205).
20. **Pourquoi ThresholdOptimizer, et quelles alternatives ?** Post-traitement : pas de ré-entraînement, contrainte ciblée, appris sur la validation. Défauts : il utilise le sexe à l'inférence et ses prédictions sont randomisées. Alternatives : reweighing, ExponentiatedGradient.

### Questions pièges possibles (bonus)
- **« Une IA a-t-elle écrit le code ? »** Oui, Claude Code a assisté l'écriture et le débogage ; c'est déclaré dans le rapport. Tous les chiffres viennent de l'exécution du code. Soyez prêts à expliquer **chaque** étape sans notes.
- **« Pourquoi OpenML puis UCI ? »** OpenML renvoyait une erreur 503 ; le code se replie sur la source UCI d'origine (mêmes 48 842 lignes).
- **« Pourquoi avoir refait l'entraînement ? »** Le premier run avait un garde-fou de temps trop court : XGBoost n'avait eu qu'un essai. Seul le run complet (40 essais par modèle) est rapporté.
- **« Y a-t-il un endpoint servi ? »** Oui : `uvicorn src.api:app`, route `POST /predict` (FastAPI).

## 3. À savoir par cœur
- **Rôles de chaque ensemble :** train = apprendre + CV d'Optuna ; validation = seuil, choix du modèle, recalibration, mitigation ; test = rapport final seulement.
- **Où est quoi :** `src/train.py` (Optuna), `runs/trials_log.csv` (journal), `src/fairness.py`, `report/gen_numbers.py` (chiffres du rapport).
- **Tout relancer :** `python run_all.py` (graine 42).
