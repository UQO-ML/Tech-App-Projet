# Tech-App-Devoir-II — INF6243

Projet de classification de tweets en 3 classes:
- `hate_speech`
- `offensive_language`
- `neither`

## Objectif du dépôt

Le projet compare plusieurs familles de modèles (classiques, GPU cuML, DistilBERT) avec une orchestration multi-runs.
Le flux principal:
1. préparer/nettoyer les données;
2. entraîner plusieurs modèles;
3. évaluer (validation + test + CV);
4. sélectionner un meilleur compromis via un score pondéré;
5. générer des artefacts (figures, rapports, modèle sauvegardé).

## Structure actuelle

```text
Tech-App-Devoir-II/
├── main.py                         # point d'entrée recommandé (CLI multi-runs)
├── notebook_principal.ipynb        # workflow interactif notebook
├── requirements.txt
├── Code/
│   ├── main.py                     # pipeline unitaire (run unique)
│   ├── notebook_workflow.py        # orchestration partagée notebook/CLI
│   ├── preprocessing.py
│   ├── models.py
│   ├── run_configs.py              # matrice de runs et profils
│   ├── run_pipeline_subprocess.py  # exécution isolée par subprocess
│   ├── result_interpreter.py
│   ├── report_markdown.py
│   ├── utils.py
│   └── model_zoo/
├── Data/
│   ├── labeled_data.csv
│   └── lien_vers_dataset.txt
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.arch
│   ├── Dockerfile.fedora
│   ├── docker-compose.simple.yml
│   └── docker-compose.pipeline.yml
├── Docs/
└── Outputs/                        # créé automatiquement
```

## Prérequis

### Exécution locale (hors Docker)

- Linux/macOS (ou WSL2 sur Windows) recommandé;
- Python 3.11 recommandé;
- `pip` à jour;
- option GPU/cuML: CUDA + pilotes NVIDIA cohérents avec les libs installées.

Important: le `requirements.txt` mentionne explicitement:

```bash
unset CUDA_PATH
```

à exécuter avant installation si votre environnement exporte `CUDA_PATH` et crée des conflits de résolution.

### Exécution Docker (recommandée pour GPU)

- Docker + plugin Compose;
- pilote NVIDIA compatible;
- NVIDIA Container Toolkit configuré.

Test rapide:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 nvidia-smi
```

## Installation locale

Depuis la racine du projet:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
unset CUDA_PATH
pip install -r requirements.txt
```

Vérification minimale:

```bash
python -c "import sklearn, pandas, matplotlib; print('OK')"
```

## How-to: exécuter le projet

## 1) Utiliser `main.py` (CLI, recommandé)

Le `main.py` à la racine est le lanceur principal multi-runs.

### Commandes utiles

Run rapide:

```bash
python main.py --run-matrix default
```

Run complet:

```bash
python main.py --run-matrix exhaustive
```

Avec ajustement du malus DistilBERT CV proxy:

```bash
python main.py --run-matrix default --distilbert-proxy-penalty 0.02
```

### Ce que fait ce lanceur

- charge la matrice de runs (`default` ou `exhaustive`) depuis `Code/run_configs.py`;
- filtre les runs incompatibles avec l'environnement courant;
- exécute chaque run dans un subprocess isolé (meilleure stabilité mémoire);
- agrège et classe les runs;
- affiche un résumé du meilleur run;
- produit les artefacts dans `Outputs/`.

### Différence avec `Code/main.py`

- `main.py` (racine): orchestration multi-runs + comparaison globale.
- `Code/main.py`: pipeline unitaire (`run_pipeline`) pour un run unique (utile surtout en dev interne).

## 2) Utiliser `notebook_principal.ipynb`

Le notebook est l'interface interactive pour piloter les mêmes runs que la CLI.

### Étapes d'utilisation

1. Ouvrir `notebook_principal.ipynb`.
2. Exécuter la cellule d'import.
3. Ajuster les paramètres dans la section "Paramètres du notebook":
   - `RUN_MATRIX`;
   - `SELECTED_MODELS`;
   - profils DistilBERT/MLP/AdaBoost/GPU;
   - `DISTILBERT_PROXY_PENALTY`.
4. Exécuter la cellule d'orchestration (`run_all_configs`).
5. Lire les tableaux et le rapport synthèse.
6. Consulter les figures générées dans `Outputs/figures/`.

### Conseils pratiques notebook

- pour itérer vite: `RUN_MATRIX="default"` + peu de modèles;
- pour une analyse complète: `RUN_MATRIX="exhaustive"` + profils ciblés;
- si mémoire/VRAM limitée: éviter profils agressifs (`gpu_aggressive`, `vram_max`);
- si DistilBERT/cuML absent: runs incompatibles automatiquement ignorés.

## 3) Utiliser Docker

### Mode interactif notebook/shell

```bash
docker compose -f docker/docker-compose.simple.yml build
docker compose -f docker/docker-compose.simple.yml up notebook
```

Puis:
- URL: `http://localhost:8888`
- token par défaut: `techapp` (modifiable via `JUPYTER_TOKEN`)

Shell interactif dans le même environnement:

```bash
docker compose -f docker/docker-compose.simple.yml run --rm shell
```

### Mode pipeline non-interactif (run puis exit)

```bash
docker compose -f docker/docker-compose.pipeline.yml build
docker compose -f docker/docker-compose.pipeline.yml run --rm pipeline
```

Ce mode lance automatiquement:

```bash
python main.py --run-matrix default
```

avec limites mémoire/CPU définies dans `docker/docker-compose.pipeline.yml`.

### Variantes de build

Build alternatif Arch/Fedora:

```bash
docker build -f docker/Dockerfile.arch -t tech-app-devoir-ii:arch .
docker build -f docker/Dockerfile.fedora -t tech-app-devoir-ii:fedora .
```

Build latest plutôt que lock:

```bash
REQUIREMENTS_MODE=latest docker compose -f docker/docker-compose.simple.yml build
```

Activation RAPIDS/cuML (expérimental):

```bash
ENABLE_RAPIDS=1 docker compose -f docker/docker-compose.simple.yml build
```

## Paramètres importants (CLI + notebook)

- `RUN_MATRIX`: `default` (rapide) ou `exhaustive` (coûteux);
- `DISTILBERT_PROXY_PENALTY`: malus appliqué aux runs DistilBERT en CV proxy;
- `MODEL_PARAM_OVERRIDES["DistilBERT"]`: `epochs`, `batch_size`, `max_length`, etc.;
- `MODEL_GRID_OVERRIDES["<ModelName>"]`: surcharge de grille GridSearchCV;
- profils activables dans `Code/run_configs.py` (DistilBERT, MLP, AdaBoost, GPU);
- runs centralisés dans `Code/run_configs.py`, partagés par notebook + CLI.

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

Le notebook inclut aussi un interpréteur de résultats (`Code/result_interpreter.py`) qui imprime:
- le meilleur modèle et son score global;
- la répartition des statuts d'exécution de tous les modèles;
- un top-3 des modèles entraînés;
- les modèles à améliorer selon un seuil simple de `f1_macro`.

## Exécution multi-runs en subprocess

Les runs de `run_all_configs()` sont exécutés dans des subprocess Python isolés
via `Code/run_pipeline_subprocess.py`. Cette stratégie limite les pics mémoire
car chaque process libère ses ressources à la fin du run.

## Dépannage rapide

- Erreur Jupyter token: vérifier `JUPYTER_TOKEN` et le port exposé (`JUPYTER_PORT`);
- DistilBERT indisponible: vérifier installation de `torch`, `transformers`, `datasets`;
- modèles GPU absents: vérifier cuML/cupy + compatibilité CUDA/driver;
- exécution trop lente: commencer avec `--run-matrix default` et/ou réduire les profils actifs;
- mémoire saturée: réduire batch size DistilBERT, désactiver profils agressifs, garder l'exécution en subprocess.
