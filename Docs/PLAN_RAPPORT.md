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
- Modèles classiques: Naive Bayes, Logistic Regression, Linear SVC, KNN, Decision Tree, Random Forest, MLPClassifier.
- Modèle deep learning: DistilBERT (fine-tuning supervisé, si dépendances disponibles).
- Hyperparamètres:
  - GridSearchCV pour les modèles classiques;
  - réglages contrôlés pour DistilBERT (epochs, max_length, batch sizes).
- Métriques: accuracy, précision macro, rappel macro, F1 macro, confusion matrix.
- Validation croisée: k-fold pour les modèles compatibles sklearn; fallback validation documenté pour DistilBERT (coût computationnel).
- Critère de sélection final: score pondéré (validation/test/CV) + règle de tie-break.

### 5.1 Configuration expérimentale (notebook)
- Détailler les constantes utilisées:
  - `MAX_SAMPLES`, `DISTILBERT_EPOCHS`, `INCLUDE_DISTILBERT`;
  - `TEST_SIZE`, `VAL_SIZE`, `CV_FOLDS`;
  - `SCORING`, `SELECTION_WEIGHTS`, `RANDOM_STATE`;
  - `DISTILBERT_PROXY_PENALTY` (règle de compensation DistilBERT CV proxy).
- Expliquer l'impact attendu de chaque constante sur temps d'exécution et qualité des résultats.

### 5.2 Stratégie multi-runs (A/B/C)
- **Run A — data_balance**:
  - objectif: baseline robuste en maximisant la couverture de données (`max_samples=None`);
  - intérêt: mieux observer l'effet sur la classe minoritaire `hate_speech`.
- **Run B — classic_focus**:
  - objectif: isoler la performance des modèles classiques (`include_distilbert=False`);
  - intérêt: comparer des approches homogènes avec CV complète et coût réduit.
- **Run C — distilbert_focus**:
  - objectif: tester une hypothèse deep learning différente (`distilbert_epochs` plus élevé);
  - intérêt: mesurer si plus d'entraînement améliore le rappel/F1 macro sans confondre les facteurs.
- Justifier pourquoi ces runs sont différents:
  - chaque run modifie un levier principal;
  - cela évite des conclusions ambiguës dues à trop de changements simultanés.

## 6. Résultats et discussion
- Tableau comparatif complet des scores (validation + test + CV moyen) pour tous les modèles.
- Tableau de statut d'exécution (trained/skipped/failed) avec causes (`error_or_reason`).
- Tableau de comparaison inter-runs (`metrics_report_run_*.json`) avec:
  - meilleur modèle par run;
  - score de sélection du meilleur et score ajusté (`adjusted_selection_score`);
  - F1 macro test du meilleur.
- Figure `models_comparison_test.png`.
- Figure `models_compilation_overview.png`.
- Figure `models_status_overview.png`.
- Matrices de confusion.
- Figure `confusion_matrices_all_models.png`.
- Figure `runs_comparison_overview.png` (échelle zoomée entre 0.6 et 0.8).
- Analyse d’erreurs: classes confondues et explications possibles.
- Sortie de l'interpréteur du notebook:
  - top-3 des modèles,
  - diagnostic de généralisation (écart validation vs test),
  - diagnostic de stabilité (écart CV vs test),
  - recommandations de tuning.
- Discussion: points forts/faibles des modèles.

## 7. Conclusion
- Meilleur modèle retenu et justification.
- Limites actuelles.
- Améliorations futures (optimisation DistilBERT, calibration des seuils, équilibrage avancé, augmentation de données).

## 8. Références
- Articles/docs utilisés.
- Documentation sklearn.
- Source du dataset.
