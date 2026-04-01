# Plan PowerPoint (10 diapositives)

Utilise ce plan pour `Presentation_INF6243_NomEtudiants.pptx`.

## Slide 1 — Titre
- Projet INF6243
- Sujet: détection de langage haineux/offensant
- Équipe

## Slide 2 — Problématique
- Pourquoi ce problème est important
- Objectif de classification en 3 classes

## Slide 3 — Dataset
- Source, taille, variables principales
- Exemple de tweets (avec avertissement contenu offensant)

## Slide 4 — EDA
- Distribution des classes
- Valeurs manquantes
- Corrélations (variables numériques)

## Slide 5 — Prétraitement
- Nettoyage texte
- TF-IDF
- Split train/val/test
- Paramétrage expérimental via constantes du notebook (`RUN_CONFIG`)
- Introduire la logique **multi-runs** (A/B/C) et le principe "un levier principal par run"

## Slide 6 — Modèles testés
- Modèles classiques: Naive Bayes, Logistic Regression, Linear SVC, KNN, Decision Tree, Random Forest, MLPClassifier
- Modèle deep learning: DistilBERT
- Hyperparamètres: GridSearchCV (classiques) + réglages DistilBERT (epochs)

## Slide 7 — Résultats globaux
- Graphique comparatif des modèles
- Choix de la métrique principale (F1 macro)
- Figure de compilation globale (`models_compilation_overview.png`)
- Couverture de tous les modèles et statuts (`models_status_overview.png`)
- Tableau comparatif des meilleurs résultats par run (`metrics_report_run_*.json`)

## Slide 8 — Analyse détaillée
- Matrice de confusion du meilleur modèle
- Matrices de confusion de tous les modèles (`confusion_matrices_all_models.png`)
- Exemples d’erreurs typiques
- Interpréteur de résultats du notebook (diagnostic + recommandations)

## Slide 9 — Modèle final
- Meilleur modèle retenu
- Validation croisée (k-fold) et fallback DistilBERT expliqué
- Termes influents (feature importance)

## Slide 10 — Conclusion et perspectives
- Résumé en 3 points
- Leçons de la stratégie multi-runs:
  - ce qui gagne avec plus de données (Run A),
  - ce qui gagne sans deep learning (Run B),
  - ce qui gagne avec focus DistilBERT (Run C)
- Améliorations futures
- Questions
