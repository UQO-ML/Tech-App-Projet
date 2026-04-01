# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `RandomForest`
- Score de sélection: `0.7212`
- F1 macro test du meilleur: `0.7250`
- Échantillons: `12000`

## Statuts d'exécution
- trained: `8`
- skipped: `0`
- failed: `0`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean`
- Poids: validation=0.3500, test=0.4000, cv=0.2500
- Modèles avec CV proxy: `['DistilBERT']`

## Configuration du run
- max_samples: `12000`
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
| NaiveBayes | trained | 0.5058 | 0.5229 | 0.4978 | 0.4948 |  |
| LogisticRegression | trained | 0.7110 | 0.6954 | 0.7273 | 0.7069 |  |
| LinearSVC | trained | 0.7142 | 0.7091 | 0.7209 | 0.7106 |  |
| KNN | trained | 0.5222 | 0.5368 | 0.5438 | 0.4671 |  |
| DecisionTree | trained | 0.6763 | 0.6613 | 0.6858 | 0.6820 |  |
| RandomForest | trained | 0.7212 | 0.7220 | 0.7250 | 0.7138 |  |
| MLPClassifier | trained | 0.5513 | 0.5442 | 0.5498 | 0.5637 |  |
| DistilBERT | trained | 0.6169 | 0.6117 | 0.6247 | 0.6117 |  |
