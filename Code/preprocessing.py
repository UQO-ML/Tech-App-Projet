"""
preprocessing.py — Prétraitement des données pour le projet INF 6243.

Rôle :
  - Chargement du dataset depuis un fichier local (Data/) ou une URL (p.ex. lue depuis lien_vers_dataset.txt).
  - Analyse exploratoire : statistiques descriptives, types, valeurs manquantes, corrélations, distribution des classes.
  - Nettoyage : doublons, colonnes vides ou non informatives, valeurs aberrantes (outliers) selon le contexte.
  - Gestion des valeurs manquantes : suppression (lignes/colonnes) ou imputation (moyenne, médiane, mode).
  - Encodage des variables catégorielles (LabelEncoder pour la cible, OneHotEncoder ou LabelEncoder pour les features).
  - Mise à l’échelle des variables numériques (StandardScaler ou MinMaxScaler) pour les algorithmes sensibles (SVM, KNN, etc.).
  - Séparation train / validation / test de façon stratifiée pour préserver les proportions de classes.

Structure détaillée :
  1. Imports (pandas, numpy, pathlib, sklearn.model_selection, sklearn.preprocessing, sklearn.impute)
  2. load_data : lecture selon extension/URL, retour DataFrame
  3. exploratory_summary : affichage structuré pour le rapport (shape, describe, manquants, corrélations, compte par classe)
  4. clean_data : drop_duplicates, suppression colonnes à trop de manquants ou sans variance ; optionnel : seuils d’outliers
  5. handle_missing : stratégie drop (lignes ou colonnes) ou impute (SimpleImputer avec stratégie par type)
  6. encode_features : OneHotEncoder pour features catégorielles (éviter fuite de données : fit sur train uniquement)
  7. scale_features : fit sur train, transform sur train/val/test avec le même scaler pour cohérence
  8. train_val_test_split : deux appels à train_test_split stratifié (train+val vs test, puis train vs val)
  9. get_preprocessing_pipeline (optionnel) : Pipeline sklearn pour enchaîner imputation, encodage, scaling
"""

# -----------------------------------------------------------------------------
# 1. Imports
# -----------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
# from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------------
# 2. Chargement
# -----------------------------------------------------------------------------
# def load_data(path_or_url):
#     """
#     Charge le dataset et retourne un DataFrame. path_or_url : str ou Path (fichier local) ou URL.
#     Adapter selon le sujet : pd.read_csv pour csv, pd.read_parquet pour parquet, requests+io pour URL.
#     Gérer les encodages (encoding="utf-8" ou "latin-1") et séparateurs si nécessaire.
#     """
#     pass

# -----------------------------------------------------------------------------
# 3. Résumé exploratoire
# -----------------------------------------------------------------------------
# def exploratory_summary(df, target_column=None):
#     """
#     Affiche (ou retourne) un résumé pour l’EDA : df.shape, df.dtypes, df.describe(), nombre de manquants
#     par colonne (isna().sum()), et si target_column est fourni la distribution des classes (value_counts).
#     Optionnel : matrice de corrélation des colonnes numériques. Utile pour le rapport et le choix du prétraitement.
#     """
#     pass

# -----------------------------------------------------------------------------
# 4. Nettoyage
# -----------------------------------------------------------------------------
# def clean_data(df, drop_duplicates=True, drop_empty_columns=True, outlier_method=None):
#     """
#     Retourne un DataFrame nettoyé. drop_duplicates : supprimer les lignes dupliquées. drop_empty_columns :
#     supprimer les colonnes entièrement vides ou à variance nulle. outlier_method : None, "iqr" ou "zscore" pour
#     supprimer ou capter les valeurs extrêmes (à documenter dans le rapport selon le domaine).
#     """
#     pass

# -----------------------------------------------------------------------------
# 5. Valeurs manquantes
# -----------------------------------------------------------------------------
# def handle_missing(df, strategy="drop", numerical_columns=None, categorical_columns=None):
#     """
#     strategy "drop" : supprimer les lignes contenant des NA (ou les colonnes à trop de NA, selon un seuil).
#     strategy "impute" : SimpleImputer(strategy="mean"/"median") pour les numériques, strategy="most_frequent"
#     pour les catégorielles. numerical_columns / categorical_columns : listes de noms de colonnes pour appliquer
#     la bonne stratégie par type. Retourne un DataFrame sans NA (ou avec NA imputés).
#     """
#     pass

# -----------------------------------------------------------------------------
# 6. Encodage
# -----------------------------------------------------------------------------
# def encode_features(X, categorical_columns, strategy="onehot", fit=True, encoders=None):
#     """
#     Encode les variables catégorielles pour obtenir des entiers ou des one-hot. strategy "onehot" :
#     OneHotEncoder (get_dummies ou sklearn) pour des features sans ordre. strategy "label" : LabelEncoder
#     par colonne pour des entiers. fit=True : créer et ajuster les encodeurs (sur l’ensemble d’entraînement) ;
#     fit=False : utiliser encoders fournis pour transform uniquement (validation/test). Retourne (X_encoded, encoders).
#     """
#     pass

# -----------------------------------------------------------------------------
# 7. Mise à l’échelle
# -----------------------------------------------------------------------------
# def scale_features(X, method="standard", fit=True, scaler=None):
#     """
#     method "standard" : StandardScaler (moyenne 0, écart-type 1). method "minmax" : MinMaxScaler (plage 0–1).
#     fit=True : instancier et ajuster le scaler sur X (à utiliser sur X_train). fit=False : appliquer scaler
#     existant (sur X_val, X_test) pour éviter la fuite d’information. Retourne (X_scaled, scaler).
#     """
#     pass

# -----------------------------------------------------------------------------
# 8. Split train / validation / test
# -----------------------------------------------------------------------------
# def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42, stratify=True):
#     """
#     Produit 6 sorties : X_train, X_val, X_test, y_train, y_val, y_test. stratify=y (ou True) assure que les
#     proportions de classes sont conservées dans chaque split (important en classification déséquilibrée).
#     Implémentation : d’abord split (train+val) vs test, puis split train vs val sur (train+val), avec les
#     mêmes random_state et stratify pour reproductibilité.
#     """
#     pass

# -----------------------------------------------------------------------------
# 9. Pipeline (optionnel)
# -----------------------------------------------------------------------------
# def get_preprocessing_pipeline(numerical_cols, categorical_cols, scale_method="standard"):
#     """
#     Retourne un sklearn.pipeline.Pipeline (ou ColumnTransformer) qui enchaîne imputation, encodage des
#     catégorielles, scaling des numériques. Permet de fit sur train puis transform sur train/val/test
#     sans dupliquer la logique et en évitant les fuites de données.
#     """
#     pass
