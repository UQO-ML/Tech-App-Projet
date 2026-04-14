# Projet : Detection de Hate Speech et Langage Offensif

Projet de classification de tweets en 3 classes:
- `hate_speech`
- `offensive_language`
- `neither`

Le depot inclut:
- une orchestration multi-runs (CLI + notebook),
- des modeles classiques scikit-learn,
- des variantes GPU cuML,
- un modele DistilBERT (si dependances deep learning disponibles),
- une generation automatique d'artefacts dans `Outputs/`.

## Structure du projet (livrable)

```text
Tech-App-Devoir-II/
├── main.py                      # point d'entree CLI multi-runs
├── hate-speech-model.ipynb      # notebook principal (livrable)
├── models_tests.ipynb           # notebook de tests/experiences
├── requirements.txt
├── Code/
│   ├── main.py                  # pipeline unitaire (run unique)
│   ├── notebook_workflow.py     # orchestration partagee notebook/CLI
│   ├── run_configs.py           # matrices de runs + profils
│   ├── run_pipeline_subprocess.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── report_markdown.py
│   ├── result_interpreter.py
│   ├── utils.py
│   └── model_zoo/
├── data/
│   └── labeled_data.csv
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── .env.example
│   ├── requirements.base*.txt
│   └── requirements.rapids*.txt
├── Docs/
└── Outputs/                     # cree automatiquement
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

Verification minimale:

```bash
python -c "import sklearn, pandas, matplotlib; print('OK')"
```

## Execution (CLI, recommande)

Le lanceur principal est `main.py` a la racine.

Run rapide:

```bash
python main.py --run-matrix default
```

Run complet:

```bash
python main.py --run-matrix exhaustive
```

Avec ajustement du malus DistilBERT (CV proxy):

```bash
python main.py --run-matrix exhaustive --distilbert-proxy-penalty 0.02
```

## Execution notebook

Notebook principal: `hate-speech-model.ipynb`.

```bash
jupyter lab
```

Puis ouvrir `hate-speech-model.ipynb` et executer les cellules dans l'ordre.

## Execution Docker

Le compose principal est `docker/docker-compose.yml`.

Build:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env build
```

Lancer JupyterLab:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up notebook
```

Acces:
- URL: `http://localhost:8888`
- token: `techapp` (modifiable via `JUPYTER_TOKEN` dans `docker/.env`)

## Artefacts generes

Le pipeline ecrit dans `Outputs/`:
- `Outputs/figures/` (comparaisons, matrices de confusion, courbes, importances),
- `Outputs/reports/metrics_report.json` (report principal),
- `Outputs/reports/metrics_report.md` (version lisible),
- `Outputs/reports/metrics_report_<run>.json/.md` (reports par run),
- `Outputs/reports/runs_comparison_overview.md`,
- `Outputs/reports/error_cases_best_model.json/.md`,
- `Outputs/models/best_model.joblib` (ou note deep learning selon le cas).

## Parametres importants

- `RUN_MATRIX`: `default` ou `exhaustive`.
- `DISTILBERT_PROXY_PENALTY`: malus applique aux runs DistilBERT en CV proxy.
- `SELECTED_MODELS` et profils (dans le notebook) pour filtrer les runs.
- `Code/run_configs.py`: reference unique des profils et matrices de runs.

## Depannage rapide

- DistilBERT indisponible: verifier `torch`, `transformers`, `datasets`.
- Modeles GPU absents: verifier CUDA/driver + cuML/cupy.
- Resultats incomplets dans `metrics_report.json`: c'est le report du run de reference; consulter aussi `metrics_report_run_*.json`.
- Erreurs de profils (`'u','l','t','r','a'`): utiliser des tuples, ex. `("ultra",)`.

## References

- Dataset: [Hate Speech and Offensive Language Dataset (Kaggle)](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
- scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
- transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- RAPIDS cuML: [https://docs.rapids.ai/api/cuml/stable/](https://docs.rapids.ai/api/cuml/stable/)
