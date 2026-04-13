# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `DistilBERT`
- Score de sélection: `0.7073`
- F1 macro test du meilleur: `0.7618`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `1`
- skipped: `0`
- failed: `0`
- modèles attendus: `['DistilBERT']`
- modèles entraînés: `['DistilBERT']`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test - penalty_if(hate_recall_test < hate_recall_floor)`
- Poids: validation=0.3000, test=0.4000, cv=0.1500, hate_recall=0.1500
- Seuil hate_recall: `0.4000`
- Pénalité hate_recall: `0.0300`
- Politique précision: `La précision macro est suivie comme métrique diagnostique, mais n'est pas utilisée comme critère principal de sélection.`
- Modèles avec CV proxy: `['DistilBERT']`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `2`
- include_distilbert: `True`
- algorithm_switches: `{'NaiveBayes': False, 'LogisticRegression': False, 'LinearSVC': False, 'KNN': False, 'DecisionTree': False, 'RandomForest': False, 'AdaBoost': False, 'MLPClassifier': False, 'LogisticRegressionGPU': False, 'LinearSVCGPU': False, 'KNNGPU': False, 'RandomForestGPU': False, 'DistilBERT': True}`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- model_param_overrides: `{'DistilBERT': {'epochs': 2, 'batch_size': 16, 'max_length': 160, 'learning_rate': 3e-05, 'weight_decay': 0.01}}`
- model_grid_overrides: `{}`
- selection_weights: `[0.3, 0.4, 0.15, 0.15]`
- hate_recall_floor: `0.4`
- hate_recall_penalty: `0.03`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DistilBERT | trained | 0.7073 | 0.7495 | 0.7607 | 0.7618 | 0.7607 ± n/a | 0.4021 | 0.4475 | 0.0000 |  |

## Analyse d'erreurs textuelles
- Fichier JSON: `Outputs/reports/error_cases_best_model.json`
- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`
- Résumé features par modèle: `Outputs/reports/feature_importance_summary.json`
- Heatmap comparative: `Outputs/figures/feature_importance_comparison_models.png`
