# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `DistilBERT`
- Score de sélection: `0.7400`
- F1 macro test du meilleur: `0.7543`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `8`
- skipped: `0`
- failed: `0`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean`
- Poids: validation=0.4500, test=0.4000, cv=0.1500
- Modèles avec CV proxy: `['DistilBERT']`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `2`
- include_distilbert: `True`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `3`
- scoring: `f1_macro`
- selection_weights: `[0.45, 0.4, 0.15]`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Val F1 | Test F1 | CV mean | Erreur |
|---|---:|---:|---:|---:|---:|---|
| NaiveBayes | trained | 0.5196 | 0.5191 | 0.5219 | 0.5152 |  |
| LogisticRegression | trained | 0.7287 | 0.7257 | 0.7327 | 0.7273 |  |
| LinearSVC | trained | 0.7252 | 0.7169 | 0.7338 | 0.7268 |  |
| KNN | trained | 0.3571 | 0.3574 | 0.3546 | 0.3631 |  |
| DecisionTree | trained | 0.6960 | 0.6920 | 0.7015 | 0.6933 |  |
| RandomForest | trained | 0.7334 | 0.7376 | 0.7318 | 0.7255 |  |
| MLPClassifier | trained | 0.5754 | 0.5790 | 0.5733 | 0.5703 |  |
| DistilBERT | trained | 0.7400 | 0.7305 | 0.7543 | 0.7305 |  |
