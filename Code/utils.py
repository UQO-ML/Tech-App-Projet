"""Fonctions utilitaires: métriques, figures, sauvegardes."""

from __future__ import annotations

import json
import math
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
MODEL_LABEL = "Modèle"


def ensure_dir(path: str | Path) -> Path:
    """Crée un dossier s'il n'existe pas.

    Paramètres:
        path: Chemin du dossier à créer.

    Retour:
        Objet `Path` du dossier.
    """
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_device() -> str:
    """Détecte le device de calcul.

    Retour:
        `"cuda"` si disponible, sinon `"cpu"`.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les métriques principales de classification.

    Paramètres:
        y_true: Labels de référence.
        y_pred: Labels prédits.

    Retour:
        Dictionnaire avec accuracy, precision_macro, recall_macro, f1_macro.
    """
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
    """Trace la distribution des classes puis sauvegarde la figure.

    Paramètres:
        y: Labels de classes.
        class_names: Mapping id -> nom de classe.
        save_path: Chemin de sortie PNG.
    """
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
    """Visualise les valeurs manquantes par variable.

    Paramètres:
        df: Données à analyser.
        save_path: Chemin de sortie PNG.
    """
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
    """Trace la heatmap de corrélation des variables numériques.

    Paramètres:
        df: Données à analyser.
        save_path: Chemin de sortie PNG.
    """
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
    """Trace l'histogramme des longueurs de tweets.

    Paramètres:
        df: Données contenant `tweet_length` et `class`.
        class_names: Mapping id -> nom de classe.
        save_path: Chemin de sortie PNG.
    """
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
    """Trace le boxplot du nombre de mots par classe.

    Paramètres:
        df: Données contenant `word_count` et `class`.
        class_names: Mapping id -> nom de classe.
        save_path: Chemin de sortie PNG.
    """
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
    """Trace la matrice de confusion annotée.

    Paramètres:
        y_true: Labels de référence.
        y_pred: Labels prédits.
        class_names: Mapping id -> nom de classe.
        title: Titre du graphique.
        save_path: Chemin de sortie PNG.
    """
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


def plot_confusion_matrices_grid(
    confusion_by_model: dict[str, np.ndarray],
    class_names: dict[int, str],
    save_path: str | Path,
) -> None:
    """Compile toutes les matrices de confusion dans une figure en grille.

    Paramètres:
        confusion_by_model: Dictionnaire `{modele: matrice_confusion}`.
        class_names: Mapping id -> nom de classe.
        save_path: Chemin de sortie PNG.
    """
    n_models = len(confusion_by_model)
    if n_models == 0:
        return
    n_cols = 3
    n_rows = math.ceil(n_models / n_cols)
    labels = [class_names[i] for i in sorted(class_names)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()
    for idx, (model_name, cm) in enumerate(confusion_by_model.items()):
        ax = axes_array[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vrai")

    for idx in range(n_models, len(axes_array)):
        axes_array[idx].axis("off")

    fig.suptitle("Matrices de confusion - Tous les modèles", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_models_comparison(metrics_by_model: dict[str, dict[str, float]], save_path: str | Path) -> None:
    """Trace un barplot Accuracy/F1 pour comparer les modèles.

    Paramètres:
        metrics_by_model: Dictionnaire `{modele: metriques}`.
        save_path: Chemin de sortie PNG.
    """
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


def plot_models_compilation(
    report_by_model: dict[str, dict[str, Any]],
    model_selection_scores: dict[str, float],
    save_path: str | Path,
) -> None:
    """
    Figure de synthèse comparative:
    - barplot accuracy/f1,
    - heatmap métriques macro,
    - barplot score global de sélection.

    Paramètres:
        report_by_model: Rapport détaillé par modèle.
        model_selection_scores: Score global de sélection par modèle.
        save_path: Chemin de sortie PNG.
    """
    rows: list[dict[str, Any]] = []
    for model_name, payload in report_by_model.items():
        test_metrics = payload.get("test_metrics", {})
        rows.append(
            {
                "model": model_name,
                "accuracy": test_metrics.get("accuracy", 0.0),
                "precision_macro": test_metrics.get("precision_macro", 0.0),
                "recall_macro": test_metrics.get("recall_macro", 0.0),
                "f1_macro": test_metrics.get("f1_macro", 0.0),
            }
        )
    frame = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    melted = frame.melt(
        id_vars="model",
        value_vars=["accuracy", "f1_macro"],
        var_name="metric",
        value_name="score",
    )

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy vs F1 macro (test)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_xlabel(MODEL_LABEL)
    axes[0].set_ylabel("Score")

    heatmap_data = frame.set_index("model")[["precision_macro", "recall_macro", "f1_macro"]]
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", ax=axes[1], cbar=False, vmin=0, vmax=1)
    axes[1].set_title("Métriques macro (test)")
    axes[1].set_xlabel("Métrique")
    axes[1].set_ylabel(MODEL_LABEL)

    selection_df = pd.DataFrame(
        {
            "model": list(model_selection_scores.keys()),
            "selection_score": list(model_selection_scores.values()),
        }
    ).sort_values("selection_score", ascending=False)
    sns.barplot(data=selection_df, x="model", y="selection_score", color="#4575b4", ax=axes[2])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Score global de sélection")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_xlabel(MODEL_LABEL)
    axes[2].set_ylabel("Score")

    fig.suptitle("Compilation comparative de tous les modèles", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_status_overview(
    all_models_report: dict[str, dict[str, Any]],
    save_path: str | Path,
) -> None:
    """Trace un aperçu des statuts d'exécution pour tous les modèles attendus.

    Paramètres:
        all_models_report: Dictionnaire complet du report par modèle.
        save_path: Chemin de sortie PNG.
    """
    rows = []
    for model_name, payload in all_models_report.items():
        rows.append(
            {
                "model": model_name,
                "status": payload.get("status", "unknown"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["status_code"] = frame["status"].map({"trained": 1, "skipped": 0, "failed": -1}).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.countplot(data=frame, x="status", order=["trained", "skipped", "failed"], ax=axes[0])
    axes[0].set_title("Nombre de modèles par statut")
    axes[0].set_xlabel("Statut")
    axes[0].set_ylabel("Nombre")

    sns.barplot(data=frame, x="model", y="status_code", hue="status", dodge=False, ax=axes[1])
    axes[1].set_title("Statut de chaque modèle")
    axes[1].set_xlabel(MODEL_LABEL)
    axes[1].set_ylabel("Code statut")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["failed", "skipped", "trained"])
    axes[1].tick_params(axis="x", rotation=25)
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    fig.suptitle("Couverture de tous les modèles", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_runs_comparison(run_summary: pd.DataFrame, save_path: str | Path) -> None:
    """Trace une comparaison inter-runs sur score brut/ajusté et F1 test.

    Paramètres:
        run_summary: DataFrame contenant `run`, `best_selection_score`, `best_test_f1_macro`
            et optionnellement `adjusted_selection_score`.
        save_path: Chemin de sortie PNG.
    """
    if run_summary.empty:
        return

    frame = run_summary.copy()
    if "adjusted_selection_score" in frame.columns:
        frame = frame.sort_values("adjusted_selection_score", ascending=False)
        comparison_metrics = ["best_selection_score", "adjusted_selection_score", "best_test_f1_macro"]
        ranking_metric = "adjusted_selection_score"
    else:
        frame = frame.sort_values("best_selection_score", ascending=False)
        comparison_metrics = ["best_selection_score", "best_test_f1_macro"]
        ranking_metric = "best_selection_score"

    melted = frame.melt(
        id_vars="run",
        value_vars=comparison_metrics,
        var_name="metric",
        value_name="score",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=melted, x="run", y="score", hue="metric", ax=axes[0])
    axes[0].set_ylim(0.6, 0.8)
    axes[0].set_title("Comparaison des runs")
    axes[0].set_xlabel("Run")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=20)

    rank = frame[["run", "best_model"]].reset_index(drop=True)
    rank["order"] = rank.index + 1
    sns.barplot(data=frame, x="run", y=ranking_metric, color="#4575b4", ax=axes[1])
    axes[1].set_ylim(0.6, 0.8)
    axes[1].set_title("Classement inter-runs (ordre décroissant)")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel(ranking_metric)
    axes[1].tick_params(axis="x", rotation=20)
    for idx, row in rank.iterrows():
        axes[1].text(idx, row["order"] * 0 + frame.iloc[idx][ranking_metric] + 0.01, f"#{row['order']} {row['best_model']}", ha="center", fontsize=8)

    fig.suptitle("Vue comparative inter-runs", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_learning_curves(estimator: Any, X: pd.Series, y: pd.Series, save_path: str | Path) -> None:
    """Trace les courbes d'apprentissage du modèle fourni.

    Paramètres:
        estimator: Modèle sklearn entraînable.
        X: Features.
        y: Labels.
        save_path: Chemin de sortie PNG.
    """
    if getattr(estimator, "skip_cv", False):
        plt.figure(figsize=(8, 4))
        plt.text(
            0.5,
            0.5,
            "Courbe d'apprentissage non tracée pour ce modèle\n(cross-validation trop coûteuse / non compatible).",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

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
    """Trace l'importance des features d'un pipeline si disponible.

    Paramètres:
        model_pipeline: Pipeline sklearn avec étapes `tfidf` et `clf`.
        save_path: Chemin de sortie PNG.
        top_k: Nombre maximal de features à afficher.

    Retour:
        `True` si une figure a été créée, sinon `False`.
    """
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
    """Sauvegarde un modèle sklearn/pipeline avec joblib.

    Paramètres:
        model: Modèle à sauvegarder.
        name: Nom de fichier sans extension.

    Retour:
        Chemin du fichier `.joblib`.
    """
    ensure_dir(MODELS_DIR)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Sauvegarde un dictionnaire JSON lisible (indenté, UTF-8).

    Paramètres:
        payload: Données à sérialiser.
        path: Chemin de sortie JSON.

    Retour:
        Chemin du fichier JSON écrit.
    """
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return out_path


def build_classification_report(y_true: pd.Series, y_pred: np.ndarray, class_names: dict[int, str]) -> dict[str, Any]:
    """Construit un rapport de classification structuré.

    Paramètres:
        y_true: Labels de référence.
        y_pred: Labels prédits.
        class_names: Mapping id -> nom de classe.

    Retour:
        Dictionnaire compatible JSON (sortie `classification_report`).
    """
    labels = sorted(class_names.keys())
    target_names = [class_names[i] for i in labels]
    return classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
