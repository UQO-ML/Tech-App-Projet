# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `LogisticRegression`
- Score de sélection: `0.6977`
- F1 macro test du meilleur: `0.7327`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `3`
- skipped: `0`
- failed: `0`
- modèles attendus: `['LogisticRegression', 'RandomForest', 'DistilBERT']`
- modèles entraînés: `['LogisticRegression', 'RandomForest', 'DistilBERT']`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test - penalty_if(hate_recall_test < hate_recall_floor)`
- Poids: validation=0.3000, test=0.3500, cv=0.2000, hate_recall=0.1500
- Seuil hate_recall: `0.4000`
- Pénalité hate_recall: `0.0300`
- Politique précision: `La précision macro est suivie comme métrique diagnostique, mais n'est pas utilisée comme critère principal de sélection.`
- Modèles avec CV proxy: `['DistilBERT']`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `1`
- include_distilbert: `True`
- algorithm_switches: `{'NaiveBayes': False, 'LogisticRegression': True, 'LinearSVC': False, 'KNN': False, 'DecisionTree': False, 'RandomForest': True, 'AdaBoost': False, 'MLPClassifier': False, 'LogisticRegressionGPU': False, 'LinearSVCGPU': False, 'KNNGPU': False, 'RandomForestGPU': False, 'DistilBERT': True}`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- model_param_overrides: `{}`
- model_grid_overrides: `{}`
- selection_weights: `[0.3, 0.35, 0.2, 0.15]`
- hate_recall_floor: `0.4`
- hate_recall_penalty: `0.03`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LogisticRegression | trained | 0.6977 | 0.7748 | 0.7257 | 0.7327 | 0.7298 ± 0.0055 | 0.5175 | 0.4398 | 0.0000 |  |
| RandomForest | trained | 0.6887 | 0.7612 | 0.7342 | 0.7251 | 0.7299 ± 0.0092 | 0.4580 | 0.4302 | 0.0000 |  |
| DistilBERT | trained | 0.5970 | 0.6883 | 0.7033 | 0.7028 | 0.7033 ± n/a | 0.1958 | 0.2779 | 0.0300 |  |

## Analyse d'erreurs textuelles
- Fichier JSON: `Outputs/reports/error_cases_best_model.json`
- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`
- Résumé features par modèle: `Outputs/reports/feature_importance_summary.json`
- Heatmap comparative: `Outputs/figures/feature_importance_comparison_models.png`
