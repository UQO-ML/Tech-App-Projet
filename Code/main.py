"""
main.py — Point d’entrée du pipeline INF 6243.

Enchaîne tout le flux : chargement des données → analyse exploratoire → prétraitement
→ entraînement des modèles (≥4 algorithmes) → évaluation sur test → visualisations
→ sélection et sauvegarde du meilleur modèle. Si des réseaux de neurones sont utilisés,
le device (CUDA en priorité, repli CPU) est déterminé au démarrage via utils.get_device()
et réutilisé pour tous les modèles GPU.

Structure détaillée (à implémenter) :
  1. Imports et configuration : chemins (ROOT, DATA_DIR), RANDOM_STATE, optionnellement affichage du device
  2. Chargement : lecture du dataset depuis Data/ ou URL (lien_vers_dataset.txt)
  3. Exploration : résumé statistique, valeurs manquantes, distribution des classes, figures EDA
  4. Prétraitement : nettoyage, gestion des manquants, encodage, mise à l’échelle, split train/val/test
  5. Entraînement : appel à models.train_all_models (ou boucle sur train_with_grid_search) ; pour un NN, passer device
  6. Évaluation sur test : métriques et matrices de confusion pour chaque modèle
  7. Visualisations : comparaison des modèles, courbes d’apprentissage, importance des features
  8. Meilleur modèle : choix justifié et sauvegarde (utils.save_model)

Exécution : depuis la racine du projet « python main.py » ou « python Code/main.py ».
"""

# -----------------------------------------------------------------------------
# 1. Imports et configuration
# -----------------------------------------------------------------------------
# import sys
# from pathlib import Path
# ROOT = Path(__file__).resolve().parent.parent   # Racine du projet (Tech-App-Devoir-II)
# DATA_DIR = ROOT / "Data"   # Dossier des données ; lire aussi Data/lien_vers_dataset.txt pour l’URL
# sys.path.insert(0, str(Path(__file__).resolve().parent))   # Pour importer preprocessing, models, utils
# import preprocessing as prep
# import models
# import utils
# RANDOM_STATE = 42   # Utiliser partout pour reproductibilité (split, seeds)
# # Optionnel : device = utils.get_device() ; print(f"Using device: {device}")   # CUDA si dispo, sinon CPU

# -----------------------------------------------------------------------------
# 2. Chargement des données
# -----------------------------------------------------------------------------
# Construire data_path (fichier local dans Data/ ou URL lue depuis lien_vers_dataset.txt).
# prep.load_data(data_path) doit retourner un DataFrame (adapter selon le format du sujet : csv, etc.).
# data_path = DATA_DIR / "votre_fichier.csv"
# df = prep.load_data(data_path)

# -----------------------------------------------------------------------------
# 3. Exploration (EDA)
# -----------------------------------------------------------------------------
# prep.exploratory_summary(df) : shape, dtypes, describe(), valeurs manquantes, distribution de la cible.
# Ensuite : visualisations (histogrammes des variables numériques, boxplots, heatmap de corrélations,
# distribution des classes). Sauvegarder les figures dans utils.FIGURES_DIR si défini.

# -----------------------------------------------------------------------------
# 4. Prétraitement
# -----------------------------------------------------------------------------
# Nettoyage (doublons, colonnes inutiles, outliers si pertinent), puis gestion des manquants (drop ou impute).
# Séparer X (features) et y (cible). Encoder les variables catégorielles, normaliser/standardiser si besoin.
# Split stratifié train / validation / test avec prep.train_val_test_split (pour garder les proportions de classes).
# df_clean = prep.clean_data(df)
# df_clean = prep.handle_missing(df_clean, strategy="...")
# X, y = ..., ...
# X_train, X_val, X_test, y_train, y_val, y_test = prep.train_val_test_split(X, y, ...)

# -----------------------------------------------------------------------------
# 5. Entraînement
# -----------------------------------------------------------------------------
# Appel à models.train_all_models(X_train, y_train, X_val, y_val) pour lancer la recherche d’hyperparamètres
# et l’évaluation sur la validation pour chaque modèle. Si vous avez un NN PyTorch, passer device=get_device()
# pour que le modèle et les tenseurs utilisent CUDA en priorité, avec repli sur CPU.
# results = models.train_all_models(X_train, y_train, X_val, y_val)

# -----------------------------------------------------------------------------
# 6. Évaluation sur test
# -----------------------------------------------------------------------------
# Pour chaque modèle dans results : prédire sur X_test, calculer les métriques (utils.compute_metrics),
# afficher/sauvegarder la matrice de confusion (utils.plot_confusion_matrix). Construire metrics_by_model
# pour l’étape de comparaison.
# for name, res in results.items():
#     y_pred = res["estimator"].predict(X_test)
#     utils.plot_confusion_matrix(y_test, y_pred, title=name, ...)

# -----------------------------------------------------------------------------
# 7. Visualisations comparatives
# -----------------------------------------------------------------------------
# Barplot de comparaison des modèles (ex. F1 ou accuracy) avec utils.plot_models_comparison.
# Courbes d’apprentissage du meilleur modèle avec utils.plot_learning_curves.
# Si le meilleur modèle a des feature_importances_ : utils.plot_feature_importance.
# utils.plot_models_comparison(metrics_by_model)
# utils.plot_learning_curves(best_estimator, X_train, y_train)
# utils.plot_feature_importance(best_estimator, feature_names)

# -----------------------------------------------------------------------------
# 8. Meilleur modèle
# -----------------------------------------------------------------------------
# Choisir le modèle aux meilleures performances (ex. F1 sur validation/test), justifier brièvement,
# sauvegarder avec utils.save_model(best_estimator, "meilleur_modele") pour réutilisation ultérieure.
# utils.save_model(best_estimator, "meilleur_modele")


def main():
    """Orchestre le pipeline complet : config, chargement, EDA, prétraitement, entraînement, évaluation, visualisations, sauvegarde."""
    pass


if __name__ == "__main__":
    main()
