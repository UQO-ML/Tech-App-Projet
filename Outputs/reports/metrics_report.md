# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `LogisticRegression`
- Score de sélection: `0.6977`
- F1 macro test du meilleur: `0.7327`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `4`
- skipped: `0`
- failed: `0`
- modèles attendus: `['LogisticRegression', 'RandomForest', 'MLPClassifier', 'LogisticRegressionGPU']`
- modèles entraînés: `['LogisticRegression', 'RandomForest', 'MLPClassifier', 'LogisticRegressionGPU']`

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
- algorithm_switches: `{'NaiveBayes': False, 'LogisticRegression': True, 'LinearSVC': False, 'KNN': False, 'DecisionTree': False, 'RandomForest': True, 'AdaBoost': False, 'MLPClassifier': True, 'LogisticRegressionGPU': True, 'LinearSVCGPU': False, 'KNNGPU': False, 'RandomForestGPU': False, 'DistilBERT': False}`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- model_param_overrides: `{}`
- model_grid_overrides: `{}`
- selection_weights: `[0.25, 0.25, 0.35, 0.15]`
- hate_recall_floor: `0.4`
- hate_recall_penalty: `0.03`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LogisticRegression | trained | 0.6977 | 0.7748 | 0.7257 | 0.7327 | 0.7298 ± 0.0055 | 0.5175 | 0.4398 | 0.0000 |  |
| RandomForest | trained | 0.6890 | 0.7612 | 0.7342 | 0.7251 | 0.7299 ± 0.0092 | 0.4580 | 0.4302 | 0.0000 |  |
| MLPClassifier | trained | 0.4927 | 0.6044 | 0.6018 | 0.6024 | 0.5959 ± 0.0162 | 0.0874 | 0.1515 | 0.0300 |  |
| LogisticRegressionGPU | trained | 0.5443 | 0.6396 | 0.6491 | 0.6521 | 0.6471 ± 0.0068 | 0.1503 | 0.2324 | 0.0300 |  |

## Analyse d'erreurs textuelles
- Fichier JSON: `Outputs/reports/error_cases_best_model.json`
- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`
- Résumé features par modèle: `Outputs/reports/feature_importance_summary.json`
- Heatmap comparative: `Outputs/figures/feature_importance_comparison_models.png`
