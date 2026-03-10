"""
models.py — Définition et entraînement des classificateurs pour le projet INF 6243.

Rôle :
  - Définir au moins 4 algorithmes de classification (ex. KNN, arbre de décision,
    Random Forest, SVM, régression logistique, Naive Bayes, réseau de neurones).
  - Pour chaque modèle : réglage des hyperparamètres (GridSearchCV ou RandomizedSearchCV).
  - Entraînement et retour des modèles entraînés + métriques de validation.
  - Pour l’apprentissage profond : utiliser le device fourni par utils.get_device() (CUDA en priorité,
    repli sur CPU) pour placer le modèle et les tenseurs sur le bon device.

Structure détaillée :
  1. Imports (sklearn.* ; optionnel : torch, utils.get_device)
  2. Grilles d’hyperparamètres par modèle (étendue adaptée au temps de calcul et à la taille des données)
  3. get_models() : dict { "NomAffiché": (estimator, param_grid ou None) } pour alimenter la recherche
  4. train_with_grid_search : GridSearchCV/RandomizedSearchCV, retour best_estimator + cv_results_
  5. train_all_models : boucle sur les modèles, évaluation sur validation, agrégation des résultats
  6. Réseau de neurones (optionnel) : build_nn_model + train_nn en utilisant device (CUDA si dispo, sinon CPU)
"""

# -----------------------------------------------------------------------------
# 1. Imports
# -----------------------------------------------------------------------------
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB  # ou MultinomialNB pour texte
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix
# # Pour deep learning : import torch ; from utils import get_device  # device = get_device() -> "cuda" ou "cpu"

# -----------------------------------------------------------------------------
# 2. Grilles d’hyperparamètres (à adapter au dataset et au temps de calcul)
# -----------------------------------------------------------------------------
# Chaque clé correspond à un nom de modèle ; la valeur est le param_grid (ou param_distributions pour RandomizedSearchCV).
# Réduire les listes si les essais prennent trop de temps ; augmenter n_iter pour RandomizedSearchCV si besoin.
# PARAM_GRIDS = {
#     "KNN": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
#     "DecisionTree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
#     "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
#     "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
#     "LogisticRegression": {"C": [0.1, 1, 10], "max_iter": [500, 1000]},
#     "NaiveBayes": {},  # peu d’hyperparamètres à tuner
# }

# -----------------------------------------------------------------------------
# 3. Dictionnaire des modèles (estimator + grille)
# -----------------------------------------------------------------------------
# def get_models():
#     """
#     Retourne un dict : { "NomAffiché": (estimator, param_grid ou None) }.
#     estimateur : instance non entraînée (ex. RandomForestClassifier()). param_grid : dict pour GridSearchCV ;
#     None ou {} pour les modèles sans recherche (ex. Naive Bayes). Utilisé par train_all_models et train_with_grid_search.
#     """
#     pass

# -----------------------------------------------------------------------------
# 4. Entraînement avec recherche d’hyperparamètres
# -----------------------------------------------------------------------------
# def train_with_grid_search(X_train, y_train, model_name, cv=5, scoring="f1_macro", n_jobs=-1):
#     """
#     Instancie GridSearchCV (ou RandomizedSearchCV si l’espace est grand) avec la grille du modèle model_name,
#     fit sur (X_train, y_train), retourne best_estimator_ et cv_results_. scoring : métrique à maximiser
#     (f1_macro, accuracy, etc.). n_jobs=-1 utilise tous les cœurs CPU (les modèles sklearn sont sur CPU).
#     """
#     pass

# -----------------------------------------------------------------------------
# 5. Entraîner tous les modèles
# -----------------------------------------------------------------------------
# def train_all_models(X_train, y_train, X_val, y_val, model_names=None):
#     """
#     Pour chaque modèle dans get_models() (ou dans model_names si fourni) : lance train_with_grid_search,
#     évalue le best_estimator sur (X_val, y_val) avec les métriques choisies. Retourne un dict :
#     { "nom": {"estimator": best_estimator, "val_metrics": {...}, "cv_results": cv_results_ } }.
#     """
#     pass

# -----------------------------------------------------------------------------
# 6. Réseau de neurones (optionnel) — Device CUDA en priorité, repli CPU
# -----------------------------------------------------------------------------
# def build_nn_model(input_dim, n_classes, hidden=(64, 32), dropout=0.2, device=None):
#     """
#     Construit un MLP (PyTorch nn.Module ou sklearn.neural_network.MLPClassifier). Si PyTorch : passer
#     device = get_device() (depuis utils) pour que le modèle soit créé / déplacé sur GPU si disponible.
#     Ex. : model = MyMLP(...).to(device). Tous les tenseurs d’entrée doivent aussi être envoyés sur device
#     (ex. x = torch.tensor(...).to(device)) pour que l’entraînement utilise CUDA quand c’est possible.
#     """
#     # if device is None:
#     #     from utils import get_device
#     #     device = get_device()
#     # ... model.to(device)
#     pass

# def train_nn(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, device=None):
#     """
#     Boucle d’entraînement (PyTorch) : à chaque batch, envoyer inputs et labels sur device (ex. .to(device)),
#     forward, loss, backward, optimizer.step(). À chaque epoch, évaluer sur X_val (également sur device).
#     device doit être celui retourné par get_device() : CUDA en priorité, sinon CPU, pour garantir le repli
#     automatique si le GPU est indisponible.
#     """
#     pass
