# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `RandomForest`
- Score de sélection: `0.7309`
- F1 macro test du meilleur: `0.7251`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `7`
- skipped: `0`
- failed: `0`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean`
- Poids: validation=0.4000, test=0.1500, cv=0.4500
- Modèles avec CV proxy: `[]`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `1`
- include_distilbert: `False`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- selection_weights: `[0.4, 0.15, 0.45]`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Val F1 | Test F1 | CV mean | Erreur |
|---|---:|---:|---:|---:|---:|---|
| NaiveBayes | trained | 0.5202 | 0.5191 | 0.5219 | 0.5205 |  |
| LogisticRegression | trained | 0.7286 | 0.7257 | 0.7327 | 0.7298 |  |
| LinearSVC | trained | 0.7261 | 0.7169 | 0.7338 | 0.7316 |  |
| KNN | trained | 0.3675 | 0.3668 | 0.3669 | 0.3682 |  |
| DecisionTree | trained | 0.6933 | 0.6953 | 0.6942 | 0.6912 |  |
| RandomForest | trained | 0.7309 | 0.7342 | 0.7251 | 0.7299 |  |
| MLPClassifier | trained | 0.5992 | 0.6018 | 0.6024 | 0.5959 |  |
