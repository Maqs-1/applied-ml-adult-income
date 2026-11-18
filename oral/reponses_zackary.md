# Zackary — Questions 11 à 20

**11. Pourquoi la PR-AUC est-elle plus élevée sur le test ?**

On obtient 0,845 sur le test et 0,826 en validation croisée. Les deux chiffres viennent de données et de protocoles différents. Le test est resté à part pendant les choix du modèle. Il faudrait quantifier la variabilité de la PR-AUC avant de conclure à une différence.

**12. Comment expliquer SHAP ?**

Je pars d'un score de référence. SHAP estime ce que chaque variable apporte en moyenne quand on la fait intervenir dans différents ordres. Pour notre XGBoost, la référence et les contributions s'additionnent sur l'échelle des log-odds.

**13. Quelles variables ressortent ?**

Le statut marital, les gains en capital, les études, l'âge et les heures de travail ressortent. Le fait que `relationship_Husband` arrive haut suggère que le modèle peut retrouver indirectement une partie de l'information sur le sexe. Il faudrait mesurer cet effet pour le confirmer.

**14. Pourquoi deux variables liées au mariage ressortent ?**

`is_married` et `Married-civ-spouse` encodent une information très proche. Le modèle et SHAP peuvent répartir le signal entre elles. Je les lis ensemble pour comprendre le rôle du mariage.

**15. Le modèle surestime-t-il les positifs ?**

Oui, en moyenne sur le test. Il annonce 34,5 % alors qu'on observe 23,9 % de positifs. L'ECE brute est de 0,106 et le Brier de 0,105.

**16. Utiliserais-tu les probabilités brutes ?**

Pour classer les personnes selon leur score, oui. Si j'ai besoin d'interpréter 30 % comme une fréquence, je préfère les probabilités recalibrées. La correction isotonique ajustée sur la validation fait passer l'ECE de 0,106 à 0,011 et le Brier de 0,105 à 0,084 sur le test. La pondération des classes peut contribuer au décalage initial.

**17. Que mesurent le Brier et l'ECE ?**

Le Brier fait la moyenne de `(probabilité − résultat réel)²`. Pour l'ECE, on répartit les probabilités dans des bacs de largeur égale, puis on compare la probabilité moyenne à la fréquence observée dans chaque bac. On prend la valeur absolue de l'écart et on pondère par le nombre d'exemples. La courbe de fiabilité construit ses bacs par quantiles.

**18. Parité démographique ou equalized odds ?**

La parité démographique vise le même taux de décisions positives dans chaque groupe. Avec equalized odds, je compare les taux de vrais positifs entre groupes, puis les taux de faux positifs. Les fréquences de revenu supérieur à 50 k$ varient selon le sexe dans ces données. Notre choix porte sur les écarts d'erreur à classe réelle donnée.

**19. Qu'a coûté la mitigation ?**

L'écart d'equalized odds baisse de 0,106 à 0,009 environ, mais le F1 global passe de 0,741 à 0,681. Chez les femmes, la précision passe de 0,699 à 0,335 et le taux de faux positifs monte de 0,037 à 0,205. La réduction de l'écart s'accompagne donc de davantage de faux positifs dans ce groupe.

**20. Pourquoi ThresholdOptimizer ?**

On peut ajuster la règle de décision sur la validation avec la contrainte choisie, tout en gardant le modèle déjà entraîné. La règle a besoin du sexe au moment de prédire et peut randomiser ses décisions. On aurait aussi pu modifier les poids des exemples à l'entraînement ou utiliser un algorithme avec contrainte d'équité comme ExponentiatedGradient.
