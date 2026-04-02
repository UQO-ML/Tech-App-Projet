# Rapport de métriques (lisible)

## Résumé global
- Modèle retenu: `LogisticRegression`
- Score de sélection: `0.6967`
- F1 macro test du meilleur: `0.7327`
- Échantillons: `24783`

## Statuts d'exécution
- trained: `8`
- skipped: `0`
- failed: `0`

## Méthode de sélection
- Formule: `selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test - penalty_if(hate_recall_test < hate_recall_floor)`
- Poids: validation=0.4500, test=0.2000, cv=0.2000, hate_recall=0.1500
- Seuil hate_recall: `0.4000`
- Pénalité hate_recall: `0.0300`
- Politique précision: `La précision macro est suivie comme métrique diagnostique, mais n'est pas utilisée comme critère principal de sélection.`
- Modèles avec CV proxy: `[]`

## Configuration du run
- max_samples: `None`
- distilbert_epochs: `1`
- include_distilbert: `False`
- algorithm_switches: `{'NaiveBayes': True, 'LogisticRegression': True, 'LinearSVC': True, 'KNN': True, 'DecisionTree': True, 'RandomForest': True, 'AdaBoost': True, 'MLPClassifier': True, 'DistilBERT': False}`
- test_size: `0.2`
- val_size: `0.1`
- cv_folds: `5`
- scoring: `f1_macro`
- selection_weights: `[0.45, 0.2, 0.2, 0.15]`
- hate_recall_floor: `0.4`
- hate_recall_penalty: `0.03`
- random_state: `42`

## Détail par modèle

| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NaiveBayes | trained | 0.4157 | 0.4919 | 0.5191 | 0.5219 | 0.5205 ± 0.0099 | 0.0245 | 0.0476 | 0.0300 |  |
| LogisticRegression | trained | 0.6967 | 0.7748 | 0.7257 | 0.7327 | 0.7298 ± 0.0055 | 0.5175 | 0.4398 | 0.0000 |  |
| LinearSVC | trained | 0.6792 | 0.7508 | 0.7169 | 0.7338 | 0.7316 ± 0.0069 | 0.4231 | 0.4130 | 0.0000 |  |
| KNN | trained | 0.3209 | 0.3945 | 0.3668 | 0.3669 | 0.3682 ± 0.0334 | 0.2587 | 0.1152 | 0.0300 |  |
| DecisionTree | trained | 0.6524 | 0.7425 | 0.6953 | 0.6942 | 0.6912 ± 0.0071 | 0.4161 | 0.3371 | 0.0000 |  |
| RandomForest | trained | 0.6901 | 0.7612 | 0.7342 | 0.7251 | 0.7299 ± 0.0092 | 0.4580 | 0.4302 | 0.0000 |  |
| AdaBoost | trained | 0.4506 | 0.6172 | 0.5546 | 0.5382 | 0.5224 ± 0.0262 | 0.1259 | 0.1865 | 0.0300 |  |
| MLPClassifier | trained | 0.4936 | 0.6044 | 0.6018 | 0.6024 | 0.5959 ± 0.0162 | 0.0874 | 0.1515 | 0.0300 |  |

## Analyse d'erreurs textuelles
- Fichier JSON: `Outputs/reports/error_cases_best_model.json`
- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`
