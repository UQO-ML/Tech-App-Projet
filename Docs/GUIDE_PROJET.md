# Guide rapide du projet

Ce dossier explique **quoi exécuter**, **dans quel ordre**, et **pourquoi**.

## 1) Idée générale

Le projet classe des tweets dans 3 classes:
- `0` : discours haineux (`hate_speech`)
- `1` : langage offensant (`offensive_language`)
- `2` : ni haineux ni offensant (`neither`)

Le pipeline complet est dans les scripts Python du dossier `Code/`, puis appelé depuis le notebook racine `notebook_principal.ipynb`.

## 2) Qui fait quoi ?

- `Code/preprocessing.py`
  - charge les données;
  - nettoie les tweets;
  - crée des variables EDA (`tweet_length`, `word_count`);
  - prépare le split `train/validation/test`.
- `Code/models.py`
  - définit plusieurs modèles pertinents pour texte (`NaiveBayes`, `LogisticRegression`, `LinearSVC`, `KNN`, `DecisionTree`, `RandomForest`, `AdaBoost`, `MLPClassifier`, `DistilBERT`);
  - intègre aussi 4 variantes GPU (`LogisticRegressionGPU`, `LinearSVCGPU`, `KNNGPU`, `RandomForestGPU`) basées sur cuML;
  - applique GridSearchCV pour régler les hyperparamètres;
  - applique aussi un fine-tuning DistilBERT (si dépendances deep installées);
  - calcule un score global de sélection pour retenir le meilleur modèle.
- `Code/utils.py`
  - calcule les métriques (`accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`);
  - génère les figures demandées;
  - sauvegarde les modèles et rapports JSON.
- `Code/result_interpreter.py`
  - contient l'interpréteur de résultats (diagnostic par modèle, statuts, recommandations);
  - est appelé directement depuis le notebook pour éviter de garder cette logique en cellule.
- `Code/report_markdown.py`
  - génère des versions Markdown lisibles des reports JSON (`metrics_report*.md`, `runs_comparison_overview.md`).
- `Code/run_pipeline_subprocess.py`
  - exécute un run pipeline dans un process Python isolé;
  - sert de worker pour limiter l'empreinte mémoire entre runs.
- `Code/run_configs.py`
  - centralise les matrices de runs pour le notebook et la CLI;
  - définit les profils de tuning explicites (DistilBERT, MLP, AdaBoost et modèles GPU);
  - filtre les runs incompatibles (dépendances absentes) avant exécution.
- `Code/main.py`
  - orchestre toutes les étapes;
  - produit les sorties dans `Outputs/`.

## 3) Pourquoi ce design ?

- **Scripts séparés**: plus simple à maintenir et à expliquer dans le rapport.
- **Notebook léger**: sert de vitrine de résultats, sans logique complexe cachée.
- **TF-IDF + modèles classiques**: baseline solide, rapide à entraîner, facile à justifier.
- **`f1_macro`**: pertinent pour le multi-classes et les classes potentiellement déséquilibrées.
- **Sélection robuste**: combinaison validation + test + validation croisée pour éviter un choix basé sur un seul split.

## 4) Exécution

Depuis la racine du projet:

```bash
python main.py
```

Ou dans le notebook, exécuter les cellules de haut en bas.

### Exécution Docker (recommandée pour environnement GPU durable)

Deux compositions sont disponibles dans `docker/`:
- `docker-compose.simple.yml`: notebook Jupyter + shell manuel;
- `docker-compose.pipeline.yml`: pipeline minimale puis arrêt.

Pré-requis:
- Docker + Compose;
- driver NVIDIA;
- NVIDIA Container Toolkit.

Build image:

```bash
docker compose -f docker/docker-compose.simple.yml build
```

Notebook Jupyter (conteneur GPU):

```bash
docker compose -f docker/docker-compose.simple.yml up notebook
```

Accès notebook:
- URL: `http://localhost:8888`
- token par défaut: `techapp` (surcharge via variable `JUPYTER_TOKEN`).

Shell manuel dans le conteneur:

```bash
docker compose -f docker/docker-compose.simple.yml run --rm shell
```

Exécution pipeline minimale (puis exit):

```bash
docker compose -f docker/docker-compose.pipeline.yml run --rm pipeline
```

Ce service applique des bornes mémoire/CPU et lance:
`python main.py --run-matrix default`.

Le notebook expose des constantes de pilotage, puis délègue les détails à `Code/run_configs.py`:
- `RUN_MATRIX` pour choisir la matrice (`default` ou `exhaustive`);
- `DISTILBERT_PROXY_PENALTY` pour la comparaison inter-runs;
- `DISTILBERT_PROFILES_ENABLED`, `MLP_PROFILES_ENABLED`, `ADABOOST_PROFILES_ENABLED`;
- `GPU_MODELS_ENABLED`, `GPU_PROFILES_ENABLED`.

Le notebook ne porte plus la logique détaillée de génération des runs:
- il appelle directement `get_exhaustive_runs()` / `filter_incompatible_runs()` depuis `Code/run_configs.py`;
- la maintenance des profils de tuning est donc centralisée côté scripts.

Chaque constante est commentée dans le notebook pour préciser son impact et les valeurs recommandées.
Pour les overrides, conserver des valeurs pragmatiques:
- DistilBERT `epochs`: 1-3 pour itérations courtes, 4-5 pour consolidation;
- DistilBERT `batch_size`: 8 si mémoire limitée, 16 compromis, 32 si GPU confortable;
- DistilBERT `max_length`: 128 par défaut, 160/192 si besoin de plus de contexte;
- Grilles classiques: éviter une explosion combinatoire (temps x mémoire).

Politique de scoring (sélection du meilleur modèle):
- Score de base: `w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + w_hate * hate_recall_test`.
- Garde-fou minoritaire: si `hate_recall_test < HATE_RECALL_FLOOR`, une pénalité `HATE_RECALL_PENALTY` est soustraite.
- Objectif: éviter de retenir un modèle qui optimise seulement le score global mais rate la classe `hate_speech`.
- `precision_macro` est suivie comme indicateur diagnostique, pas comme critère principal.

## 5) Sorties générées

Le pipeline crée automatiquement:
- `Outputs/figures/` : graphiques EDA, matrices de confusion, comparaison de modèles, learning curve;
- `Outputs/figures/models_compilation_overview.png` : vue comparative globale de tous les modèles;
- `Outputs/figures/models_status_overview.png` : couverture de tous les modèles attendus (trained/skipped/failed);
- `Outputs/figures/confusion_matrices_all_models.png` : compilation des matrices de confusion;
- `Outputs/figures/feature_importance_best_model.png` : termes influents du meilleur modèle;
- `Outputs/figures/feature_importance_comparison_models.png` : comparaison inter-modèles des contributions de features;
- `Outputs/reports/eda_summary.json` : résumé EDA;
- `Outputs/reports/metrics_report.json` : résultats détaillés de tous les modèles;
- `Outputs/reports/metrics_report.md` : synthèse lisible humain du report principal;
- `Outputs/reports/metrics_report_<run_name>.md` : synthèse lisible par run;
- `Outputs/reports/runs_comparison_overview.md` : synthèse lisible de la comparaison inter-runs;
- `Outputs/reports/feature_importance_summary.json` : top features influentes par modèle entraîné;
- `Outputs/reports/error_cases_best_model.json` et `.md` : exemples textuels FP/FN pour le meilleur modèle;
- `Outputs/models/best_model.joblib` : meilleur modèle sauvegardé.

Remarque DistilBERT:
- `best_cv_score` est vide (`NaN`) car DistilBERT est entraîné en fine-tuning direct (pas de GridSearchCV complet).
- Le report inclut alors un fallback de stabilité via `cv_fallback_for_models`.
- En comparaison multi-runs, ce cas peut recevoir un malus contrôlé:
  `adjusted_selection_score = best_selection_score - DISTILBERT_PROXY_PENALTY`.

Remarque GPU classiques:
- les 4 modèles GPU cuML sont évalués comme les autres via la même structure de report;
- leur backend est explicite dans `feature_config.backend = "gpu_cuml"`;
- si cuML/cupy n'est pas disponible, les runs GPU-only sont automatiquement exclus de la matrice active.

Remarque graphique:
- `runs_comparison_overview.png` utilise une échelle Y zoomée sur `[0.6, 0.8]`
  pour rendre les différences entre runs plus lisibles.

Le notebook inclut un interpréteur de résultats (via `Code/result_interpreter.py`) qui affiche un diagnostic synthétique après exécution.
