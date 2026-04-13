# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `AdaBoost`
- Score de sélection: `0.4481`
- F1 macro test du meilleur: `0.5382`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `1`
- skipped: `0`
- failed: `0`
- modèles attendus: `['AdaBoost']`
- modèles entraînés: `['AdaBoost']`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test - penalty_if(hate_recall_test < hate_recall_floor)`
- Poids: validation=0.2500, test=0.4500, cv=0.1500, hate_recall=0.1500
- Seuil hate_recall: `0.4000`
- Pénalité hate_recall: `0.0300`
- Politique précision: `La précision macro est suivie comme métrique diagnostique, mais n'est pas utilisée comme critère principal de sélection.`
- Modèles avec CV proxy: `[]`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `1`
- include_distilbert: `False`
- algorithm_switches: `{'NaiveBayes': False, 'LogisticRegression': False, 'LinearSVC': False, 'KNN': False, 'DecisionTree': False, 'RandomForest': False, 'AdaBoost': True, 'MLPClassifier': False, 'LogisticRegressionGPU': False, 'LinearSVCGPU': False, 'KNNGPU': False, 'RandomForestGPU': False, 'DistilBERT': False}`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- model_param_overrides: `{}`
- model_grid_overrides: `{'AdaBoost': {'clf__n_estimators': [200, 400, 800], 'clf__learning_rate': [0.03, 0.1, 0.3], 'clf__estimator__max_depth': [1, 2, 3], 'clf__estimator__min_samples_leaf': [1, 2]}}`
- selection_weights: `[0.25, 0.45, 0.15, 0.15]`
- hate_recall_floor: `0.4`
- hate_recall_penalty: `0.03`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AdaBoost | trained | 0.4481 | 0.6172 | 0.5546 | 0.5382 | 0.5224 ± 0.0262 | 0.1259 | 0.1865 | 0.0300 |  |

## Analyse d'erreurs textuelles
- Fichier JSON: `Outputs/reports/error_cases_best_model.json`
- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`
- Résumé features par modèle: `Outputs/reports/feature_importance_summary.json`
- Heatmap comparative: `Outputs/figures/feature_importance_comparison_models.png`
