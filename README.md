# Tech-App-Devoir-II — INF6243

Projet de classification de tweets en 3 classes:
- discours haineux (`hate_speech`)
- langage offensant (`offensive_language`)
- aucun des deux (`neither`)

## Structure actuelle

```text
Tech-App-Devoir-II/
├── main.py
├── notebook_principal.ipynb
├── requirements.txt
├── Code/
│   ├── main.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── utils.py
│   ├── result_interpreter.py
│   ├── report_markdown.py
│   ├── run_pipeline_subprocess.py
│   └── model_zoo/
├── Data/
│   ├── labeled_data.csv
│   └── lien_vers_dataset.txt
├── Docs/
│   ├── GUIDE_PROJET.md
│   ├── PLAN_RAPPORT.md
│   └── PLAN_PRESENTATION.md
└── Outputs/   # créé automatiquement à l'exécution
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Exécution

### Option 1 — Script
```bash
python main.py
```

### Option 2 — Notebook
Ouvrir `notebook_principal.ipynb` (racine du projet) et exécuter les cellules.

Dans le notebook, chaque paramètre de `RUN_CONFIG` est défini via une constante dédiée (commentée):
- `MAX_SAMPLES`: `int` (>0) ou `None` pour 100% des données;
- `DISTILBERT_EPOCHS`: `int` (ex: 1 rapide, 2-4 plus long);
- `INCLUDE_DISTILBERT`: `bool`;
- `ALGORITHM_SWITCHES`: dict `{nom_modele: bool}` pour activer/désactiver chaque algorithme;
- `TEST_SIZE` et `VAL_SIZE`: `float` entre 0 et 1;
- `CV_FOLDS`: `int` (ex: 3, 5, 10);
- `SCORING`: métrique sklearn (ex: `f1_macro`, `accuracy`);
- `MODEL_PARAM_OVERRIDES`: dict de paramètres fixes par modèle (surtout DistilBERT);
- `MODEL_GRID_OVERRIDES`: dict de surcharge de grilles GridSearch par modèle classique;
- `SELECTION_WEIGHTS`: tuple `(validation, test, cv, hate_recall)` (somme idéalement = 1.0);
- `HATE_RECALL_FLOOR`: seuil minimal de recall pour `hate_speech` sur test;
- `HATE_RECALL_PENALTY`: pénalité appliquée si le seuil n'est pas atteint;
- `RANDOM_STATE`: seed de reproductibilité.

Pour la comparaison multi-runs, une règle dédiée est aussi paramétrable:
- `DISTILBERT_PROXY_PENALTY`: `float` (recommandé: `0.00` à `0.05`) appliqué comme malus
  aux runs où DistilBERT est évalué avec CV proxy.

Guideline overrides:
- `MODEL_PARAM_OVERRIDES["DistilBERT"]["epochs"]`: 1-5 (plus grand = plus long, parfois plus performant);
- `MODEL_PARAM_OVERRIDES["DistilBERT"]["batch_size"]`: 8/16/32 (plus petit = moins de mémoire);
- `MODEL_PARAM_OVERRIDES["DistilBERT"]["max_length"]`: 96-256 (plus grand = plus de contexte, plus de coût);
- `MODEL_GRID_OVERRIDES["MLPClassifier"]`: ajouter/modifier des listes de valeurs (`clf__alpha`, `hidden_layer_sizes`, etc.) pour explorer plus large ou accélérer.

## Ce que le pipeline produit

- `Outputs/figures/` :
  - distribution des classes
  - valeurs manquantes
  - heatmap de corrélation
  - histogrammes / boxplots
  - matrices de confusion
  - compilation des matrices de confusion (tous les modèles)
  - comparaison des modèles
  - compilation synthèse comparative globale
  - couverture de statuts pour tous les modèles (trained/skipped/failed)
  - courbe d’apprentissage
  - importance des features (si supportée)
- `Outputs/reports/eda_summary.json`
- `Outputs/reports/metrics_report.json` (inclut `all_models`, statuts, erreurs éventuelles et métriques disponibles pour tous les modèles attendus)
- `Outputs/reports/metrics_report.md` (version lisible humain du report principal)
- `Outputs/reports/metrics_report_<run_name>.md` (version lisible par run)
- `Outputs/reports/runs_comparison_overview.md` (tableau lisible de comparaison inter-runs)
- `Outputs/models/best_model.joblib`
  - si meilleur modèle deep learning: `Outputs/reports/best_model_deep_learning_note.json`

## Modèles implémentés

- Naive Bayes
- Logistic Regression
- Linear SVC
- KNN
- Decision Tree
- Random Forest
- AdaBoost
- MLPClassifier
- DistilBERT (fine-tuning, si dépendances deep learning installées)

Chaque modèle est entraîné avec `GridSearchCV` et évalué avec:
- accuracy
- balanced accuracy
- précision macro
- rappel macro
- F1 macro
- F1 par classe (dont `hate_speech`)
- matrice de confusion
- validation croisée (k-fold) et IC95 (`mean ± std`) pour les modèles classiques

Sélection finale du meilleur modèle via score pondéré:
`w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test`
avec une pénalité optionnelle si `hate_recall_test` est sous `HATE_RECALL_FLOOR`.
La `precision_macro` reste un indicateur diagnostique, mais n'est pas un critère principal de sélection.

Le pipeline produit aussi une matrice d'erreurs textuelles pour le meilleur modèle:
- `Outputs/reports/error_cases_best_model.json`
- `Outputs/reports/error_cases_best_model.md`
avec des exemples de faux négatifs/faux positifs sur `hate_speech`.

Note DistilBERT:
- entraîné via fine-tuning direct (pas de GridSearchCV complet pour limiter le coût de calcul);
- si `transformers/torch/datasets` ne sont pas disponibles, le pipeline continue avec les modèles classiques.
- `best_cv_score` peut être `NaN` pour DistilBERT car il n'est pas optimisé via `GridSearchCV`; la stabilité est couverte par le fallback documenté dans `model_selection_method`.
- lors de la comparaison inter-runs, un score ajusté est utilisé:
  `adjusted_selection_score = best_selection_score - DISTILBERT_PROXY_PENALTY`
  quand DistilBERT est en mode CV proxy.

La figure `runs_comparison_overview.png` est volontairement zoomée sur l'intervalle `[0.6, 0.8]`
pour mieux visualiser les écarts fins entre runs.

Le notebook inclut aussi un **interpréteur de résultats** (script dédié `Code/result_interpreter.py`) qui imprime:
- le meilleur modèle et son score global;
- la répartition des statuts d'exécution de tous les modèles;
- un top-3 des modèles entraînés;
- les modèles à améliorer selon un seuil simple de `f1_macro`.

## Exécution multi-runs en subprocess

Les runs de `run_all_configs()` sont exécutés dans des subprocess Python isolés
via `Code/run_pipeline_subprocess.py`. Cette stratégie limite les pics mémoire
car chaque process libère ses ressources à la fin du run.
