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

## Ce que le pipeline produit

- `Outputs/figures/` :
  - distribution des classes
  - valeurs manquantes
  - heatmap de corrélation
  - histogrammes / boxplots
  - matrices de confusion
  - comparaison des modèles
  - courbe d’apprentissage
  - importance des features (si supportée)
- `Outputs/reports/eda_summary.json`
- `Outputs/reports/metrics_report.json`
- `Outputs/models/best_model.joblib`

## Modèles implémentés

- Naive Bayes
- Logistic Regression
- Linear SVC
- Random Forest

Chaque modèle est entraîné avec `GridSearchCV` et évalué avec:
- accuracy
- précision macro
- rappel macro
- F1 macro
- matrice de confusion
- validation croisée (k-fold) pour le meilleur modèle
