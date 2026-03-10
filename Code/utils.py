"""
utils.py — Fonctions utilitaires pour le projet INF 6243.

Rôle :
  - Sélection du device de calcul : CUDA en priorité, repli sur CPU si indisponible
    (utilisé par les modèles PyTorch/TensorFlow ; les classificateurs sklearn restent sur CPU).
  - Calcul des métriques (accuracy, précision, rappel, F1, matrice de confusion).
  - Visualisations : comparaison des modèles, matrices de confusion, courbes
    d’apprentissage, importance des features.
  - Helpers : chargement/sauvegarde de modèles, chemins, constantes.

Structure détaillée :
  1. Imports (numpy, pandas, matplotlib, seaborn, sklearn.metrics, pathlib ; optionnel : torch)
  2. Constantes : RANDOM_STATE (reproductibilité), FIGURES_DIR, MODELS_DIR (chemins relatifs au script)
  3. get_device() : retourne "cuda" si torch.cuda.is_available(), sinon "cpu" ; utilisé partout où
     on crée ou déplace des tenseurs/modèles PyTorch pour respecter la priorité CUDA puis repli CPU.
  4. Métriques : compute_metrics (dict avec accuracy, précision/rappel/F1 macro ou micro),
     plot_confusion_matrix (heatmap avec étiquettes de classes, option de sauvegarde).
  5. Comparaison modèles : plot_models_comparison (barplot d’une métrique par modèle, axes et légendes clairs).
  6. Courbes d’apprentissage : plot_learning_curves (learning_curve sklearn + courbe train/validation).
  7. Importance des features : plot_feature_importance (pour estimateurs avec .feature_importances_).
  8. Helpers : ensure_dir (création de dossiers), save_model/load_model (joblib ou pickle).
"""

# -----------------------------------------------------------------------------
# 1. Imports
# -----------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# from pathlib import Path
# # Pour get_device (PyTorch) : import torch

# -----------------------------------------------------------------------------
# 2. Constantes (à adapter selon le projet)
# -----------------------------------------------------------------------------
# RANDOM_STATE = 42   # Utiliser partout (train_test_split, modèles sklearn, np.random.seed) pour reproductibilité
# FIGURES_DIR = Path(__file__).resolve().parent / "figures"   # Sauvegarde des graphiques (confusion, comparaisons, etc.)
# MODELS_DIR = Path(__file__).resolve().parent / "saved_models"   # Sauvegarde des modèles entraînés

# -----------------------------------------------------------------------------
# 3. Sélection du device : CUDA en priorité, repli sur CPU
# -----------------------------------------------------------------------------
# def get_device():
#     """
#     Retourne le device à utiliser pour PyTorch : "cuda" si disponible, sinon "cpu".
#     À appeler au démarrage du pipeline (ou au premier usage d’un modèle GPU) et à réutiliser
#     pour model.to(device), tensor.to(device), etc. En cas d’absence de GPU ou de build PyTorch
#     CPU-only, torch.cuda.is_available() est False et le repli sur CPU est automatique.
#     Retourne une chaîne "cuda" ou "cpu" (ou torch.device si vous préférez).
#     """
#     # try:
#     #     import torch
#     #     return "cuda" if torch.cuda.is_available() else "cpu"
#     # except ImportError:
#     #     return "cpu"
#     pass

# -----------------------------------------------------------------------------
# 4. Métriques
# -----------------------------------------------------------------------------
# def compute_metrics(y_true, y_pred, average="macro"):
#     """
#     Retourne un dict : accuracy, precision, recall, f1 (par classe si multi-class, puis macro/micro).
#     y_true, y_pred : tableaux 1D (labels). average : "macro", "micro" ou None pour métriques par classe.
#     """
#     pass

# def plot_confusion_matrix(y_true, y_pred, labels=None, title="Matrice de confusion", save_path=None):
#     """
#     Affiche une heatmap de la matrice de confusion (seaborn.heatmap). labels : noms des classes
#     pour les axes. save_path : chemin vers un fichier (ex. FIGURES_DIR / "confusion.png") pour sauvegarder.
#     """
#     pass

# -----------------------------------------------------------------------------
# 5. Comparaison des modèles
# -----------------------------------------------------------------------------
# def plot_models_comparison(metrics_by_model, metric_name="f1", save_path=None):
#     """
#     metrics_by_model : dict { "NomModèle": { "accuracy": ..., "f1": ..., ... } }.
#     Trace un barplot (matplotlib/seaborn) de metric_name pour chaque modèle ; titre et axes étiquetés.
#     """
#     pass

# -----------------------------------------------------------------------------
# 6. Courbes d’apprentissage
# -----------------------------------------------------------------------------
# def plot_learning_curves(estimator, X, y, cv=5, train_sizes=None, save_path=None):
#     """
#     Utilise sklearn.model_selection.learning_curve pour calculer les scores train/validation
#     pour différentes tailles d’entraînement. Affiche deux courbes (train vs validation) pour
#     détecter sur/sous-apprentissage. train_sizes : liste de ratios ou d’effectifs (ex. np.linspace(0.1, 1.0, 10)).
#     """
#     pass

# -----------------------------------------------------------------------------
# 7. Importance des features (arbres, random forest, etc.)
# -----------------------------------------------------------------------------
# def plot_feature_importance(estimator, feature_names, top_k=20, save_path=None):
#     """
#     Trace un barplot horizontal des top_k plus grandes importances. Ne s’applique qu’aux estimateurs
#     ayant l’attribut feature_importances_ (DecisionTree, RandomForest, etc.). feature_names : liste
#     des noms des colonnes pour l’axe des ordonnées.
#     """
#     pass

# -----------------------------------------------------------------------------
# 8. Helpers
# -----------------------------------------------------------------------------
# def ensure_dir(path):
#     """Crée le répertoire (et les parents) si nécessaire. path : str ou Path."""
#     pass

# def save_model(model, name):
#     """Sauvegarde le modèle sous MODELS_DIR / f"{name}.joblib" (ou .pkl). Utiliser joblib pour les gros objets sklearn."""
#     pass

# def load_model(name):
#     """Charge le modèle depuis MODELS_DIR / f"{name}.joblib" (ou .pkl) et le retourne."""
#     pass
