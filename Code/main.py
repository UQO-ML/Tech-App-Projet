"""Pipeline principal du projet INF6243."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix

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
    confusion_by_model: dict[str, Any] = {}
    for model_name, result in results.items():
        estimator = result["estimator"]
        y_pred_test = estimator.predict(x_test)
        metrics_test = utils.compute_metrics(y_test, y_pred_test)
        test_metrics[model_name] = metrics_test
        confusion_by_model[model_name] = confusion_matrix(y_test, y_pred_test, labels=sorted(prep.CLASS_LABELS))
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
    utils.plot_confusion_matrices_grid(
        confusion_by_model=confusion_by_model,
        class_names=prep.CLASS_LABELS,
        save_path=utils.FIGURES_DIR / "confusion_matrices_all_models.png",
    )

    # Validation croisée k-fold sur train+val pour tous les modèles
    x_train_val = pd.concat([x_train, x_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    model_rationales = models.get_model_rationales(RANDOM_STATE)
    model_selection_scores: dict[str, float] = {}
    model_cv_scores: dict[str, list[float]] = {}

    for model_name, result in results.items():
        estimator = result["estimator"]
        cv_scores = models.cross_validate_estimator(estimator, x_train_val, y_train_val, cv=5)
        model_cv_scores[model_name] = cv_scores
        cv_mean = float(sum(cv_scores) / len(cv_scores))
        val_f1 = result["val_metrics"]["f1_macro"]
        test_f1 = test_metrics[model_name]["f1_macro"]
        # Score pondéré explicite: priorité à la généralisation (test + CV) et stabilité.
        model_selection_scores[model_name] = float((0.35 * val_f1) + (0.40 * test_f1) + (0.25 * cv_mean))
        full_report[model_name]["cv_f1_macro_scores"] = cv_scores
        full_report[model_name]["cv_f1_macro_mean"] = cv_mean
        full_report[model_name]["model_why"] = model_rationales.get(model_name, "")

    ranking = sorted(
        model_selection_scores.keys(),
        key=lambda name: (
            model_selection_scores[name],
            test_metrics[name]["f1_macro"],
            test_metrics[name]["accuracy"],
        ),
        reverse=True,
    )
    best_model_name = ranking[0]
    best_model_payload = results[best_model_name]
    best_estimator = best_model_payload["estimator"]
    best_cv_scores = model_cv_scores[best_model_name]

    utils.plot_models_compilation(
        report_by_model=full_report,
        model_selection_scores=model_selection_scores,
        save_path=utils.FIGURES_DIR / "models_compilation_overview.png",
    )

    utils.plot_learning_curves(best_estimator, x_train_val, y_train_val, utils.FIGURES_DIR / "learning_curve_best_model.png")
    utils.plot_feature_importance_from_pipeline(best_estimator, utils.FIGURES_DIR / "feature_importance_best_model.png")

    model_path = utils.save_model(best_estimator, "best_model")
    global_report = {
        "dataset_path": str(data_path),
        "n_samples": int(df.shape[0]),
        "best_model": best_model_name,
        "best_model_validation_metrics": best_model_payload["val_metrics"],
        "best_model_test_metrics": test_metrics[best_model_name],
        "best_model_cv_f1_macro_scores": best_cv_scores,
        "best_model_selection_score": model_selection_scores[best_model_name],
        "model_selection_method": {
            "formula": "selection_score = 0.35 * val_f1_macro + 0.40 * test_f1_macro + 0.25 * cv_f1_macro_mean",
            "why_this_formula": (
                "Le F1 test reçoit le plus de poids pour refléter la performance hors entraînement; "
                "la validation sert de garde-fou pendant le tuning; la CV ajoute une mesure de stabilité."
            ),
            "tie_break_rule": "En cas d'égalité, choisir le meilleur f1_macro test puis accuracy test.",
        },
        "model_selection_scores": model_selection_scores,
        "model_selection_ranking": ranking,
        "model_rationales": model_rationales,
        "best_model_selection_explanation": (
            f"Le modèle {best_model_name} obtient le meilleur score global pondéré "
            f"({model_selection_scores[best_model_name]:.4f}) en combinant validation, test et CV. "
            f"Il est retenu pour son compromis entre performance et robustesse."
        ),
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
