# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `KNNGPU`
- Score de sélection: `0.4216`
- F1 macro test du meilleur: `0.5047`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `1`
- skipped: `0`
- failed: `0`
- modèles attendus: `['KNNGPU']`
- modèles entraînés: `['KNNGPU']`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test - penalty_if(hate_recall_test < hate_recall_floor)`
- Poids: validation=0.2500, test=0.2500, cv=0.3500, hate_recall=0.1500
- Seuil hate_recall: `0.4000`
- Pénalité hate_recall: `0.0300`
- Politique précision: `La précision macro est suivie comme métrique diagnostique, mais n'est pas utilisée comme critère principal de sélection.`
- Modèles avec CV proxy: `[]`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `1`
- include_distilbert: `False`
- algorithm_switches: `{'NaiveBayes': False, 'LogisticRegression': False, 'LinearSVC': False, 'KNN': False, 'DecisionTree': False, 'RandomForest': False, 'AdaBoost': False, 'MLPClassifier': False, 'LogisticRegressionGPU': False, 'LinearSVCGPU': False, 'KNNGPU': True, 'RandomForestGPU': False, 'DistilBERT': False}`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- model_param_overrides: `{}`
- model_grid_overrides: `{'KNNGPU': {'svd__n_components': [64], 'clf__n_neighbors': [11], 'clf__weights': ['uniform']}}`
- selection_weights: `[0.25, 0.25, 0.35, 0.15]`
- hate_recall_floor: `0.4`
- hate_recall_penalty: `0.03`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| KNNGPU | trained | 0.4216 | 0.5131 | 0.5299 | 0.5047 | 0.5139 ± 0.0137 | 0.0874 | 0.1168 | 0.0300 |  |

## Analyse d'erreurs textuelles
- Fichier JSON: `Outputs/reports/error_cases_best_model.json`
- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`
- Résumé features par modèle: `Outputs/reports/feature_importance_summary.json`
- Heatmap comparative: `Outputs/figures/feature_importance_comparison_models.png`
