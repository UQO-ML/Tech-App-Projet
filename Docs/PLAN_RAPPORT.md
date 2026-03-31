# Plan de rapport (max 10 pages)

Utilise ce gabarit pour produire `Rapport_INF6243_NomEtudiants.pdf`.

## 1. Page de garde
- Titre du projet
- Noms et matricules
- Cours/session
- Date de soumission

## 2. Introduction
- Contexte: modération de contenu sur réseaux sociaux.
- Problème: classer automatiquement les tweets en 3 catégories.
- Objectif: comparer plusieurs algorithmes et retenir le meilleur.

## 3. Revue de littérature (court)
- Méthodes classiques NLP (TF-IDF + SVM/LogReg/NB).
- Limites: dépendance aux features et au domaine.
- Positionnement de votre approche (baseline robuste et interprétable).

## 4. Jeu de données et préparation
- Source: dataset Kaggle (voir `Data/lien_vers_dataset.txt`).
- Description des variables.
- Nettoyage appliqué (`clean_text`, suppression doublons/manquants).
- Split train/validation/test stratifié.

## 5. Méthodologie
- Représentation texte: TF-IDF.
- Modèles: Naive Bayes, Logistic Regression, Linear SVC, Random Forest.
- Hyperparamètres: GridSearchCV.
- Métriques: accuracy, précision macro, rappel macro, F1 macro, confusion matrix.
- Validation croisée: k-fold sur le meilleur modèle.

## 6. Résultats et discussion
- Tableau comparatif des scores (validation + test).
- Figure `models_comparison_test.png`.
- Matrices de confusion.
- Analyse d’erreurs: classes confondues et explications possibles.
- Discussion: points forts/faibles des modèles.

## 7. Conclusion
- Meilleur modèle retenu et justification.
- Limites actuelles.
- Améliorations futures (embeddings, deep learning, nettoyage avancé, équilibrage).

## 8. Références
- Articles/docs utilisés.
- Documentation sklearn.
- Source du dataset.
