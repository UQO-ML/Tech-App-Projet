# Tech-App-Devoir-II — INF6243

Projet de classification de tweets en 3 classes:
- discours haineux (`hate_speech`)
- langage offensant (`offensive_language`)
- aucun des deux (`neither`)

## Structure actuelle

```text
Tech-App-Devoir-II/
├── main.py
├── requirements.txt
├── Code/
│   ├── main.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── utils.py
│   └── notebook_principal.ipynb
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
Ouvrir `Code/notebook_principal.ipynb` et exécuter les cellules.

Dans le notebook, la cellule `RUN_CONFIG` permet de régler:
- `max_samples`
- `distilbert_epochs`

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
  - courbe d’apprentissage
  - importance des features (si supportée)
- `Outputs/reports/eda_summary.json`
- `Outputs/reports/metrics_report.json`
- `Outputs/models/best_model.joblib`
  - si meilleur modèle deep learning: `Outputs/reports/best_model_deep_learning_note.json`

## Modèles implémentés

- Naive Bayes
- Logistic Regression
- Linear SVC
- KNN
- Decision Tree
- Random Forest
- DistilBERT (fine-tuning, si dépendances deep learning installées)

Chaque modèle est entraîné avec `GridSearchCV` et évalué avec:
- accuracy
- précision macro
- rappel macro
- F1 macro
- matrice de confusion
- validation croisée (k-fold) pour le meilleur modèle

Sélection finale du meilleur modèle via score pondéré:
`0.35 * val_f1_macro + 0.40 * test_f1_macro + 0.25 * cv_f1_macro_mean`.

Note DistilBERT:
- entraîné via fine-tuning direct (pas de GridSearchCV complet pour limiter le coût de calcul);
- si `transformers/torch/datasets` ne sont pas disponibles, le pipeline continue avec les modèles classiques.
- `best_cv_score` peut être `NaN` pour DistilBERT car il n'est pas optimisé via `GridSearchCV`; la stabilité est couverte par le fallback documenté dans `model_selection_method`.
