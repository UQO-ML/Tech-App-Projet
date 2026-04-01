# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `LogisticRegression`
- Score de sélection: `0.7295`
- F1 macro test du meilleur: `0.7327`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `8`
- skipped: `0`
- failed: `0`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean`
- Poids: validation=0.3500, test=0.4000, cv=0.2500
- Modèles avec CV proxy: `['DistilBERT']`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `1`
- include_distilbert: `True`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- selection_weights: `[0.35, 0.4, 0.25]`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Val F1 | Test F1 | CV mean | Erreur |
|---|---:|---:|---:|---:|---:|---|
| NaiveBayes | trained | 0.5206 | 0.5191 | 0.5219 | 0.5205 |  |
| LogisticRegression | trained | 0.7295 | 0.7257 | 0.7327 | 0.7298 |  |
| LinearSVC | trained | 0.7274 | 0.7169 | 0.7338 | 0.7316 |  |
| KNN | trained | 0.3672 | 0.3668 | 0.3669 | 0.3682 |  |
| DecisionTree | trained | 0.6938 | 0.6953 | 0.6942 | 0.6912 |  |
| RandomForest | trained | 0.7295 | 0.7342 | 0.7251 | 0.7299 |  |
| MLPClassifier | trained | 0.6006 | 0.6018 | 0.6024 | 0.5959 |  |
| DistilBERT | trained | 0.7126 | 0.7180 | 0.7044 | 0.7180 |  |
