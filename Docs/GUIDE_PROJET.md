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
  - définit plusieurs modèles pertinents pour texte (`NaiveBayes`, `LogisticRegression`, `LinearSVC`, `KNN`, `DecisionTree`, `RandomForest`, `DistilBERT`);
  - applique GridSearchCV pour régler les hyperparamètres;
  - applique aussi un fine-tuning DistilBERT (si dépendances deep installées);
  - calcule un score global de sélection pour retenir le meilleur modèle.
- `Code/utils.py`
  - calcule les métriques (`accuracy`, `precision`, `recall`, `f1`);
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

Le notebook contient des constantes (une par paramètre) pour ajuster facilement:
- `MAX_SAMPLES`, `DISTILBERT_EPOCHS`, `INCLUDE_DISTILBERT`;
- `TEST_SIZE`, `VAL_SIZE`, `CV_FOLDS`;
- `SCORING`, `SELECTION_WEIGHTS`, `RANDOM_STATE`.
- `DISTILBERT_PROXY_PENALTY` pour la comparaison inter-runs.

Chaque constante est commentée dans le notebook pour préciser son impact et les valeurs recommandées.

## 5) Sorties générées

Le pipeline crée automatiquement:
- `Outputs/figures/` : graphiques EDA, matrices de confusion, comparaison de modèles, learning curve;
- `Outputs/figures/models_compilation_overview.png` : vue comparative globale de tous les modèles;
- `Outputs/figures/models_status_overview.png` : couverture de tous les modèles attendus (trained/skipped/failed);
- `Outputs/figures/confusion_matrices_all_models.png` : compilation des matrices de confusion;
- `Outputs/reports/eda_summary.json` : résumé EDA;
- `Outputs/reports/metrics_report.json` : résultats détaillés de tous les modèles;
- `Outputs/reports/metrics_report.md` : synthèse lisible humain du report principal;
- `Outputs/reports/metrics_report_<run_name>.md` : synthèse lisible par run;
- `Outputs/reports/runs_comparison_overview.md` : synthèse lisible de la comparaison inter-runs;
- `Outputs/models/best_model.joblib` : meilleur modèle sauvegardé.

Remarque DistilBERT:
- `best_cv_score` est vide (`NaN`) car DistilBERT est entraîné en fine-tuning direct (pas de GridSearchCV complet).
- Le report inclut alors un fallback de stabilité via `cv_fallback_for_models`.
- En comparaison multi-runs, ce cas peut recevoir un malus contrôlé:
  `adjusted_selection_score = best_selection_score - DISTILBERT_PROXY_PENALTY`.

Remarque graphique:
- `runs_comparison_overview.png` utilise une échelle Y zoomée sur `[0.6, 0.8]`
  pour rendre les différences entre runs plus lisibles.

Le notebook inclut un interpréteur de résultats (via `Code/result_interpreter.py`) qui affiche un diagnostic synthétique après exécution.
