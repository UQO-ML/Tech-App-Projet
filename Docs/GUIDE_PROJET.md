# Guide rapide du projet

Ce dossier explique **quoi exécuter**, **dans quel ordre**, et **pourquoi**.

## 1) Idée générale

Le projet classe des tweets dans 3 classes:
- `0` : discours haineux (`hate_speech`)
- `1` : langage offensant (`offensive_language`)
- `2` : ni haineux ni offensant (`neither`)

Le pipeline complet est dans les scripts Python du dossier `Code/`, puis appelé depuis le notebook `Code/notebook_principal.ipynb`.

## 2) Qui fait quoi ?

- `Code/preprocessing.py`
  - charge les données;
  - nettoie les tweets;
  - crée des variables EDA (`tweet_length`, `word_count`);
  - prépare le split `train/validation/test`.
- `Code/models.py`
  - définit 4 modèles de classification;
  - applique GridSearchCV pour régler les hyperparamètres;
  - sélectionne le meilleur modèle selon `f1_macro`.
- `Code/utils.py`
  - calcule les métriques (`accuracy`, `precision`, `recall`, `f1`);
  - génère les figures demandées;
  - sauvegarde les modèles et rapports JSON.
- `Code/main.py`
  - orchestre toutes les étapes;
  - produit les sorties dans `Outputs/`.

## 3) Pourquoi ce design ?

- **Scripts séparés**: plus simple à maintenir et à expliquer dans le rapport.
- **Notebook léger**: sert de vitrine de résultats, sans logique complexe cachée.
- **TF-IDF + modèles classiques**: baseline solide, rapide à entraîner, facile à justifier.
- **`f1_macro`**: pertinent pour le multi-classes et les classes potentiellement déséquilibrées.

## 4) Exécution

Depuis la racine du projet:

```bash
python main.py
```

Ou dans le notebook, exécuter les cellules de haut en bas.

## 5) Sorties générées

Le pipeline crée automatiquement:
- `Outputs/figures/` : graphiques EDA, matrices de confusion, comparaison de modèles, learning curve;
- `Outputs/reports/eda_summary.json` : résumé EDA;
- `Outputs/reports/metrics_report.json` : résultats détaillés de tous les modèles;
- `Outputs/models/best_model.joblib` : meilleur modèle sauvegardé.
