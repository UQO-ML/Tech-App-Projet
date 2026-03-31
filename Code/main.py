"""Pipeline principal du projet INF6243."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import models
import preprocessing as prep
import utils

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "labeled_data.csv"


def _prepare_outputs() -> None:
    """Crée les dossiers de sortie."""
    utils.ensure_dir(utils.OUTPUTS_DIR)
    utils.ensure_dir(utils.FIGURES_DIR)
    utils.ensure_dir(utils.MODELS_DIR)
    utils.ensure_dir(utils.REPORTS_DIR)


def run_pipeline(data_path: str | Path = DATA_PATH, max_samples: int | None = None) -> dict[str, Any]:
    """Exécute toutes les étapes demandées et retourne un résumé des artefacts."""
    _prepare_outputs()
    device = utils.get_device()
    print(f"Device détecté (pour deep learning): {device}")

    raw_df = prep.load_data(data_path)
    df = prep.clean_data(raw_df)
    if max_samples is not None and 0 < max_samples < len(df):
        # Echantillonnage stratifié pour accélérer les essais sans casser l'équilibre des classes.
        ratio = max_samples / len(df)
        df = df.groupby("class", group_keys=False).sample(frac=ratio, random_state=RANDOM_STATE).reset_index(drop=True)
    summary = prep.exploratory_summary(df, target_column="class")
    utils.save_json(summary, utils.REPORTS_DIR / "eda_summary.json")

    # Visualisations EDA
    utils.plot_class_distribution(df["class"], prep.CLASS_LABELS, utils.FIGURES_DIR / "class_distribution.png")
    utils.plot_missing_values(df, utils.FIGURES_DIR / "missing_values.png")
    utils.plot_numeric_correlation(df, utils.FIGURES_DIR / "correlation_heatmap.png")
    utils.plot_text_length(df, prep.CLASS_LABELS, utils.FIGURES_DIR / "tweet_length_histogram.png")
    utils.plot_word_count_boxplot(df, prep.CLASS_LABELS, utils.FIGURES_DIR / "word_count_boxplot.png")

    # Split train / val / test
    x = df["clean_tweet"]
    y = df["class"]
    x_train, x_val, x_test, y_train, y_val, y_test = prep.train_val_test_split(
        x,
        y,
        test_size=0.2,
        val_size=0.1,
        random_state=RANDOM_STATE,
    )

    # Entraînement des 4 modèles avec tuning
    results = models.train_all_models(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=RANDOM_STATE,
    )

    # Évaluation test + matrices de confusion
    test_metrics: dict[str, dict[str, float]] = {}
    full_report: dict[str, Any] = {}
    for model_name, result in results.items():
        estimator = result["estimator"]
        y_pred_test = estimator.predict(x_test)
        metrics_test = utils.compute_metrics(y_test, y_pred_test)
        test_metrics[model_name] = metrics_test
        full_report[model_name] = {
            "validation_metrics": result["val_metrics"],
            "test_metrics": metrics_test,
            "tuning": result["tuning"],
            "classification_report_test": utils.build_classification_report(y_test, y_pred_test, prep.CLASS_LABELS),
        }
        utils.plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred_test,
            class_names=prep.CLASS_LABELS,
            title=f"Matrice de confusion - {model_name}",
            save_path=utils.FIGURES_DIR / f"confusion_{model_name}.png",
        )

    utils.plot_models_comparison(test_metrics, utils.FIGURES_DIR / "models_comparison_test.png")

    # Sélection du meilleur modèle (validation F1 macro)
    best_model_name, best_model_payload = models.select_best_model(results, metric_name="f1_macro")
    best_estimator = best_model_payload["estimator"]

    # Validation croisée k-fold sur train+val du meilleur modèle
    x_train_val = pd.concat([x_train, x_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    cv_scores = models.cross_validate_estimator(best_estimator, x_train_val, y_train_val, cv=5)

    utils.plot_learning_curves(best_estimator, x_train_val, y_train_val, utils.FIGURES_DIR / "learning_curve_best_model.png")
    utils.plot_feature_importance_from_pipeline(best_estimator, utils.FIGURES_DIR / "feature_importance_best_model.png")

    model_path = utils.save_model(best_estimator, "best_model")
    global_report = {
        "dataset_path": str(data_path),
        "n_samples": int(df.shape[0]),
        "best_model": best_model_name,
        "best_model_validation_metrics": best_model_payload["val_metrics"],
        "best_model_test_metrics": test_metrics[best_model_name],
        "best_model_cv_f1_macro_scores": cv_scores,
        "all_models": full_report,
    }
    report_path = utils.save_json(global_report, utils.REPORTS_DIR / "metrics_report.json")

    return {
        "best_model_name": best_model_name,
        "best_model_path": str(model_path),
        "report_path": str(report_path),
        "figures_dir": str(utils.FIGURES_DIR),
        "outputs_dir": str(utils.OUTPUTS_DIR),
    }


def main() -> None:
    """Point d'entrée script pour terminal."""
    results = run_pipeline()
    print("Pipeline terminé.")
    print(f"Meilleur modèle: {results['best_model_name']}")
    print(f"Rapport JSON: {results['report_path']}")
    print(f"Figures: {results['figures_dir']}")


if __name__ == "__main__":
    main()
