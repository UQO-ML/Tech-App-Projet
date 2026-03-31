"""Fonctions utilitaires: métriques, figures, sauvegardes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import learning_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "Outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def ensure_dir(path: str | Path) -> Path:
    """Crée le dossier demandé si nécessaire."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_device() -> str:
    """Sélectionne CUDA si disponible, sinon CPU."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Retourne les métriques principales en macro moyenne."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def plot_class_distribution(y: pd.Series, class_names: dict[int, str], save_path: str | Path) -> None:
    """Affiche et sauvegarde la distribution des classes."""
    plt.figure(figsize=(8, 4))
    counts = y.value_counts().sort_index()
    labels = [class_names.get(int(i), str(i)) for i in counts.index]
    sns.barplot(x=labels, y=counts.values)
    plt.title("Distribution des classes")
    plt.xlabel("Classe")
    plt.ylabel("Nombre de tweets")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_missing_values(df: pd.DataFrame, save_path: str | Path) -> None:
    """Visualise les valeurs manquantes par variable."""
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure(figsize=(10, 4))
    if missing.empty:
        plt.text(0.5, 0.5, "Aucune valeur manquante", ha="center", va="center")
        plt.axis("off")
    else:
        sns.barplot(x=missing.index, y=missing.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Nombre de valeurs manquantes")
    plt.title("Valeurs manquantes")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_numeric_correlation(df: pd.DataFrame, save_path: str | Path) -> None:
    """Affiche la heatmap de corrélation des variables numériques."""
    numeric = df.select_dtypes(include=["number"])
    plt.figure(figsize=(8, 6))
    if numeric.shape[1] < 2:
        plt.text(0.5, 0.5, "Pas assez de variables numériques", ha="center", va="center")
        plt.axis("off")
    else:
        corr = numeric.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Corrélations (variables numériques)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_text_length(df: pd.DataFrame, class_names: dict[int, str], save_path: str | Path) -> None:
    """Histogramme des longueurs de tweet."""
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="tweet_length", hue="class", bins=40, kde=True, element="step")
    plt.title("Distribution de la longueur des tweets")
    plt.xlabel("Nombre de caractères")
    plt.ylabel("Fréquence")
    plt.legend(labels=[class_names.get(i, str(i)) for i in sorted(df["class"].unique())], title="Classe")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_word_count_boxplot(df: pd.DataFrame, class_names: dict[int, str], save_path: str | Path) -> None:
    """Boxplot du nombre de mots par classe."""
    df_plot = df.copy()
    df_plot["class_name"] = df_plot["class"].map(class_names)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_plot, x="class_name", y="word_count")
    plt.title("Nombre de mots par classe")
    plt.xlabel("Classe")
    plt.ylabel("Nombre de mots")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: dict[int, str],
    title: str,
    save_path: str | Path,
) -> None:
    """Trace la matrice de confusion annotée."""
    labels = sorted(class_names.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_names[i] for i in labels],
        yticklabels=[class_names[i] for i in labels],
    )
    plt.title(title)
    plt.xlabel("Prédiction")
    plt.ylabel("Vrai label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_models_comparison(metrics_by_model: dict[str, dict[str, float]], save_path: str | Path) -> None:
    """Barplot comparant Accuracy et F1 macro pour chaque modèle."""
    rows: list[dict[str, Any]] = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy", 0.0),
                "f1_macro": metrics.get("f1_macro", 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    long_frame = frame.melt(id_vars="model", value_vars=["accuracy", "f1_macro"], var_name="metric", value_name="score")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=long_frame, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Comparaison des modèles")
    plt.xlabel("Modèle")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_learning_curves(estimator: Any, X: pd.Series, y: pd.Series, save_path: str | Path) -> None:
    """Courbes d'apprentissage du meilleur modèle."""
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        shuffle=True,
        random_state=42,
        train_sizes=np.linspace(0.2, 1.0, 5),
    )
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_mean, marker="o", label="Train")
    plt.plot(train_sizes, valid_mean, marker="o", label="Validation")
    plt.title("Courbes d'apprentissage (F1 macro)")
    plt.xlabel("Taille du jeu d'entraînement")
    plt.ylabel("Score F1 macro")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance_from_pipeline(
    model_pipeline: Any,
    save_path: str | Path,
    top_k: int = 20,
) -> bool:
    """Trace les termes importants si le classifieur le permet. Retourne True si figure créée."""
    if not hasattr(model_pipeline, "named_steps"):
        return False

    tfidf = model_pipeline.named_steps.get("tfidf")
    clf = model_pipeline.named_steps.get("clf")
    if tfidf is None or clf is None or not hasattr(tfidf, "get_feature_names_out"):
        return False

    feature_names = tfidf.get_feature_names_out()
    importances = None

    if hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).mean(axis=0)
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_

    if importances is None:
        return False

    top_idx = np.argsort(importances)[-top_k:]
    plt.figure(figsize=(9, 6))
    plt.barh(feature_names[top_idx], importances[top_idx])
    plt.title("Top mots influents du meilleur modèle")
    plt.xlabel("Importance")
    plt.ylabel("Terme")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def save_model(model: Any, name: str) -> Path:
    """Sauvegarde un modèle sklearn/pipeline avec joblib."""
    ensure_dir(MODELS_DIR)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Sauvegarde un dictionnaire JSON de manière lisible."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return out_path


def build_classification_report(y_true: pd.Series, y_pred: np.ndarray, class_names: dict[int, str]) -> dict[str, Any]:
    """Génère un rapport de classification dict (facile à enregistrer en JSON)."""
    labels = sorted(class_names.keys())
    target_names = [class_names[i] for i in labels]
    return classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
