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
│   ├── run_configs.py
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

## Exécution Docker (GPU)

Des fichiers prêts à l'emploi sont fournis dans `docker/`:
- `docker/docker-compose.simple.yml`: usage interactif (Jupyter + shell manuel);
- `docker/docker-compose.pipeline.yml`: exécution pipeline minimale puis arrêt.
- `docker/Dockerfile`: image CUDA/NVIDIA (recommandée pour GPU).
- `docker/Dockerfile.arch`: variante base Arch Linux.
- `docker/Dockerfile.fedora`: variante base Fedora (compatible écosystème Red Hat).

### Prérequis machine

- Docker + plugin Compose;
- driver NVIDIA compatible;
- NVIDIA Container Toolkit configuré pour Docker.

Test rapide GPU Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 nvidia-smi
```

Build manuel d'une variante de Dockerfile:

```bash
docker build -f docker/Dockerfile.arch -t tech-app-devoir-ii:arch .
docker build -f docker/Dockerfile.fedora -t tech-app-devoir-ii:fedora .
```

Avec requirements figés (défaut) ou latest:

```bash
docker build -f docker/Dockerfile.arch --build-arg REQUIREMENTS_MODE=freeze -t tech-app-devoir-ii:arch .
docker build -f docker/Dockerfile.fedora --build-arg REQUIREMENTS_MODE=latest -t tech-app-devoir-ii:fedora .
```

Note versioning Docker:
- `docker/requirements.base.txt` et `docker/requirements.rapids.txt` sont en mode *rolling latest* (pas de pin), donc les versions les plus récentes disponibles sont installées au moment du build;
- des versions figées sont aussi fournies:
  - `docker/requirements.base.lock.txt`
  - `docker/requirements.rapids.lock.txt`
  et le `docker/Dockerfile` les utilise par défaut (`REQUIREMENTS_MODE=freeze`).

Build en mode figé (défaut):

```bash
docker compose -f docker/docker-compose.simple.yml build
```

Build en mode latest:

```bash
REQUIREMENTS_MODE=latest docker compose -f docker/docker-compose.simple.yml build
```

### 1) Mode simple: notebook ou shell manuel

Depuis la racine du projet:

```bash
docker compose -f docker/docker-compose.simple.yml build
docker compose -f docker/docker-compose.simple.yml up notebook
```

Puis ouvrir Jupyter:
- URL: `http://localhost:8888`
- token par défaut: `techapp` (modifiable via `JUPYTER_TOKEN`).

Pour un shell interactif dans le même environnement conteneurisé:

```bash
docker compose -f docker/docker-compose.simple.yml run --rm shell
```

Exemple dans ce shell:

```bash
python main.py --run-matrix default
```

### 2) Mode pipeline minimal (run puis exit)

```bash
docker compose -f docker/docker-compose.pipeline.yml build
docker compose -f docker/docker-compose.pipeline.yml run --rm pipeline
```

Ce mode:
- utilise le GPU;
- applique des bornes mémoire/CPU via Compose;
- lance `python main.py --run-matrix default`;
- s'arrête automatiquement en fin d'exécution.

### Option RAPIDS/cuML (expérimentale)

L'image Docker installe par défaut la stack stable (`docker/requirements.base.txt`).
Pour ajouter RAPIDS/cuML au build:

```bash
ENABLE_RAPIDS=1 docker compose -f docker/docker-compose.simple.yml build
```

Même logique pour `docker-compose.pipeline.yml`.
Sur GPU très récents, cette option reste plus fragile que la stack stable.

## Exécution

### Option 1 — Script
```bash
python main.py
```

### Option 2 — Notebook
Ouvrir `notebook_principal.ipynb` (racine du projet) et exécuter les cellules.

Dans le notebook, la configuration est pilotée par quelques constantes haut niveau:
- `RUN_MATRIX`: `"default"` ou `"exhaustive"`;
- `DISTILBERT_PROXY_PENALTY`: malus appliqué aux runs avec DistilBERT en CV proxy;
- `DISTILBERT_PROFILES_ENABLED`, `MLP_PROFILES_ENABLED`, `ADABOOST_PROFILES_ENABLED`: profils activés;
- `GPU_MODELS_ENABLED`, `GPU_PROFILES_ENABLED`: activation des modèles/profils GPU.

Les détails de grilles et d'hyperparamètres sont centralisés dans `Code/run_configs.py` (source unique CLI + notebook).

Pour la comparaison multi-runs, une règle dédiée est aussi paramétrable:
- `DISTILBERT_PROXY_PENALTY`: `float` (recommandé: `0.00` à `0.05`) appliqué comme malus
  aux runs où DistilBERT est évalué avec CV proxy.

Guideline overrides:
- `MODEL_PARAM_OVERRIDES["DistilBERT"]["epochs"]`: 1-5 (plus grand = plus long, parfois plus performant);
- `MODEL_PARAM_OVERRIDES["DistilBERT"]["batch_size"]`: 8/16/32 (plus petit = moins de mémoire);
- `MODEL_PARAM_OVERRIDES["DistilBERT"]["max_length"]`: 96-256 (plus grand = plus de contexte, plus de coût);
- `MODEL_GRID_OVERRIDES["MLPClassifier"]`: ajouter/modifier des listes de valeurs (`clf__alpha`, `hidden_layer_sizes`, etc.) pour explorer plus large ou accélérer.
- `MODEL_GRID_OVERRIDES["<ModelGPU>"]`: ajuster les grilles des modèles GPU (`LogisticRegressionGPU`, `LinearSVCGPU`, `KNNGPU`, `RandomForestGPU`).

Configuration centralisée des runs:
- les matrices de runs (`default` / `exhaustive`) sont définies dans `Code/run_configs.py`;
- `main.py` et le notebook réutilisent la même source pour éviter la duplication;
- les runs incompatibles avec l'environnement courant (ex: cuML absent, DistilBERT deps absentes) sont filtrés automatiquement.

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
  - importance des features du meilleur modèle (`feature_importance_best_model.png`)
  - heatmap de contribution des features entre modèles (`feature_importance_comparison_models.png`)
- `Outputs/reports/eda_summary.json`
- `Outputs/reports/metrics_report.json` (inclut `all_models`, statuts, erreurs éventuelles et métriques disponibles pour tous les modèles attendus)
- `Outputs/reports/metrics_report.md` (version lisible humain du report principal)
- `Outputs/reports/metrics_report_<run_name>.md` (version lisible par run)
- `Outputs/reports/runs_comparison_overview.md` (tableau lisible de comparaison inter-runs)
- `Outputs/reports/feature_importance_summary.json` (top termes influents par modèle entraîné)
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
- LogisticRegressionGPU (cuML)
- LinearSVCGPU (cuML)
- KNNGPU (cuML)
- RandomForestGPU (cuML)

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

Note GPU classiques (cuML):
- les modèles GPU sont intégrés dans la pipeline comme les autres via `model_zoo`;
- si les dépendances cuML/cupy sont absentes, les runs GPU-only sont ignorés dans l'orchestration multi-runs;
- dans le report, `feature_config.backend` indique le backend (`cpu_sklearn`, `gpu_cuml`, `gpu_torch`).

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
