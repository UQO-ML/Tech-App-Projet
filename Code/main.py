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
    """Crée tous les dossiers de sortie nécessaires au pipeline."""
    utils.ensure_dir(utils.OUTPUTS_DIR)
    utils.ensure_dir(utils.FIGURES_DIR)
    utils.ensure_dir(utils.MODELS_DIR)
    utils.ensure_dir(utils.REPORTS_DIR)


def _sample_dataframe(df: pd.DataFrame, max_samples: int | None, random_state: int) -> pd.DataFrame:
    """Sous-échantillonne le dataset en conservant l'équilibre des classes."""
    if max_samples is None or not (0 < max_samples < len(df)):
        return df
    ratio = max_samples / len(df)
    return df.groupby("class", group_keys=False).sample(frac=ratio, random_state=random_state).reset_index(drop=True)


def _build_empty_model_report(status: str, error: str | None, model_why: str, tuning_info: dict[str, Any], val_metrics: dict[str, Any]) -> dict[str, Any]:
    """Construit le squelette de reporting pour un modèle (actif ou non)."""
    return {
        "status": status,
        "error": error,
        "model_why": model_why,
        "tuning": tuning_info,
        "validation_metrics": val_metrics,
        "test_metrics": {},
        "classification_report_test": {},
        "cv_f1_macro_scores": [],
        "cv_f1_macro_mean": None,
        "selection_score": None,
        "feature_config": {},
    }


def _build_feature_config(model_name: str, tuning_info: dict[str, Any]) -> dict[str, Any]:
    """Extrait les métadonnées de représentation (TF-IDF/Transformer)."""
    best_params = tuning_info.get("best_params", {}) if isinstance(tuning_info, dict) else {}
    return {
        "representation": "Transformer embeddings" if model_name == "DistilBERT" else "TF-IDF sparse vectors",
        "tfidf_ngram_range": best_params.get("tfidf__ngram_range"),
        "tfidf_min_df": best_params.get("tfidf__min_df"),
        "tfidf_max_features": best_params.get("tfidf__max_features"),
        "deep_model_name": best_params.get("model_name") if model_name == "DistilBERT" else None,
        "max_length": best_params.get("max_length") if model_name == "DistilBERT" else None,
    }


def _evaluate_models(
    expected_model_names: list[str],
    results: dict[str, dict[str, Any]],
    model_rationales: dict[str, str],
    x_test: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    """Évalue les modèles entraînés sur le test et enrichit le report par modèle."""
    test_metrics: dict[str, dict[str, float]] = {}
    full_report: dict[str, dict[str, Any]] = {}
    confusion_by_model: dict[str, Any] = {}
    trained_model_names: list[str] = []

    for model_name in expected_model_names:
        result = results.get(model_name, {})
        status = result.get("status", "skipped")
        error = result.get("error")
        tuning_info = result.get("tuning", {})
        val_metrics = result.get("val_metrics", {})

        full_report[model_name] = _build_empty_model_report(
            status=status,
            error=error,
            model_why=model_rationales.get(model_name, ""),
            tuning_info=tuning_info,
            val_metrics=val_metrics,
        )

        if status != "trained":
            continue

        estimator = result.get("estimator")
        if estimator is None:
            full_report[model_name]["status"] = "failed"
            full_report[model_name]["error"] = "Modèle marqué trained mais estimateur absent."
            continue

        trained_model_names.append(model_name)
        y_pred_test = estimator.predict(x_test)
        metrics_test = utils.compute_metrics(y_test, y_pred_test)
        test_metrics[model_name] = metrics_test
        confusion_by_model[model_name] = confusion_matrix(y_test, y_pred_test, labels=sorted(prep.CLASS_LABELS))
        full_report[model_name]["test_metrics"] = metrics_test
        full_report[model_name]["classification_report_test"] = utils.build_classification_report(y_test, y_pred_test, prep.CLASS_LABELS)
        full_report[model_name]["feature_config"] = _build_feature_config(model_name, tuning_info)
        utils.plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred_test,
            class_names=prep.CLASS_LABELS,
            title=f"Matrice de confusion - {model_name}",
            save_path=utils.FIGURES_DIR / f"confusion_{model_name}.png",
        )

    return test_metrics, full_report, confusion_by_model, trained_model_names


def _compute_selection_scores(
    trained_model_names: list[str],
    results: dict[str, dict[str, Any]],
    test_metrics: dict[str, dict[str, float]],
    x_train_val: pd.Series,
    y_train_val: pd.Series,
    full_report: dict[str, dict[str, Any]],
    cv_folds: int,
    selection_weights: tuple[float, float, float],
) -> tuple[dict[str, float], dict[str, list[float]], list[str]]:
    """Calcule CV, score pondéré de sélection et fallback CV éventuel."""
    model_selection_scores: dict[str, float] = {}
    model_cv_scores: dict[str, list[float]] = {}
    cv_fallback_models: list[str] = []

    for model_name in trained_model_names:
        result = results[model_name]
        estimator = result["estimator"]
        cv_scores = models.cross_validate_estimator(estimator, x_train_val, y_train_val, cv=cv_folds)
        if cv_scores:
            model_cv_scores[model_name] = cv_scores
            cv_mean = float(sum(cv_scores) / len(cv_scores))
        else:
            # Pour DistilBERT, la CV complète est coûteuse; on approxime avec le F1 validation.
            cv_mean = float(result["val_metrics"]["f1_macro"])
            model_cv_scores[model_name] = [cv_mean]
            cv_fallback_models.append(model_name)

        val_f1 = result["val_metrics"]["f1_macro"]
        test_f1 = test_metrics[model_name]["f1_macro"]
        w_val, w_test, w_cv = selection_weights
        model_selection_scores[model_name] = float((w_val * val_f1) + (w_test * test_f1) + (w_cv * cv_mean))
        full_report[model_name]["cv_f1_macro_scores"] = cv_scores
        full_report[model_name]["cv_f1_macro_mean"] = cv_mean
        full_report[model_name]["selection_score"] = model_selection_scores[model_name]

    return model_selection_scores, model_cv_scores, cv_fallback_models


def _build_distilbert_note(distilbert_status: str, results: dict[str, dict[str, Any]]) -> str:
    """Construit un message utilisateur lisible sur l'état DistilBERT."""
    if distilbert_status == "trained":
        return "DistilBERT inclus dans la comparaison."
    if distilbert_status == "failed":
        return f"DistilBERT en échec: {results.get('DistilBERT', {}).get('error', 'erreur inconnue')}."
    return "DistilBERT ignoré: dépendances deep learning non disponibles dans l'environnement courant."


def _save_best_model_artifact(best_model_name: str, best_estimator: Any) -> Path:
    """Sauvegarde le meilleur modèle selon son type (sklearn vs deep learning)."""
    if getattr(best_estimator, "is_deep_model", False):
        model_path = utils.REPORTS_DIR / "best_model_deep_learning_note.json"
        utils.save_json(
            {
                "best_model": best_model_name,
                "save_strategy": "manual_huggingface",
                "why": (
                    "Le meilleur modèle est un Transformer. "
                    "La sérialisation joblib n'est pas adaptée; utiliser model.save_pretrained/tokenizer.save_pretrained."
                ),
            },
            model_path,
        )
        return model_path
    return utils.save_model(best_estimator, "best_model")


def run_pipeline(
    data_path: str | Path = DATA_PATH,
    max_samples: int | None = None,
    distilbert_epochs: int = 1,
    include_distilbert: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.1,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    selection_weights: tuple[float, float, float] = (0.35, 0.40, 0.25),
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Exécute le pipeline complet de classification et retourne les artefacts produits.

    Paramètres:
        data_path: Chemin vers le fichier de données.
        max_samples: Taille maximale d'échantillon (stratifié) pour accélérer les essais.
        distilbert_epochs: Nombre d'epochs utilisé pour DistilBERT.
        include_distilbert: Active ou non DistilBERT dans la comparaison.
        test_size: Proportion du jeu de test.
        val_size: Proportion du jeu de validation.
        cv_folds: Nombre de folds pour la validation croisée.
        scoring: Métrique utilisée pour GridSearchCV.
        selection_weights: Pondérations `(validation, test, cv)` pour le score final.
        random_state: Seed globale.

    Retour:
        Dictionnaire avec le meilleur modèle, le chemin du report et les dossiers de sortie.
    """
    _prepare_outputs()
    device = utils.get_device()
    print(f"Device détecté (pour deep learning): {device}")

    raw_df = prep.load_data(data_path)
    df = prep.clean_data(raw_df)
    df = _sample_dataframe(df, max_samples, random_state=random_state)
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
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # Entraînement des modèles classiques + DistilBERT (si dépendances disponibles)
    results = models.train_all_models(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=random_state,
        include_distilbert=include_distilbert,
        distilbert_epochs=distilbert_epochs,
        cv_folds=cv_folds,
        scoring=scoring,
    )
    expected_model_names = models.get_expected_model_names(include_distilbert=include_distilbert)
    distilbert_status = results.get("DistilBERT", {}).get("status", "skipped")
    distilbert_enabled = distilbert_status == "trained"

    model_rationales = models.get_model_rationales(random_state)
    test_metrics, full_report, confusion_by_model, trained_model_names = _evaluate_models(
        expected_model_names=expected_model_names,
        results=results,
        model_rationales=model_rationales,
        x_test=x_test,
        y_test=y_test,
    )

    if test_metrics:
        utils.plot_models_comparison(test_metrics, utils.FIGURES_DIR / "models_comparison_test.png")
    if confusion_by_model:
        utils.plot_confusion_matrices_grid(
            confusion_by_model=confusion_by_model,
            class_names=prep.CLASS_LABELS,
            save_path=utils.FIGURES_DIR / "confusion_matrices_all_models.png",
        )

    # Validation croisée k-fold sur train+val pour tous les modèles
    x_train_val = pd.concat([x_train, x_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    model_selection_scores, model_cv_scores, cv_fallback_models = _compute_selection_scores(
        trained_model_names=trained_model_names,
        results=results,
        test_metrics=test_metrics,
        x_train_val=x_train_val,
        y_train_val=y_train_val,
        full_report=full_report,
        cv_folds=cv_folds,
        selection_weights=selection_weights,
    )

    if not model_selection_scores:
        raise RuntimeError("Aucun modèle n'a pu être entraîné. Vérifier les dépendances et les données.")

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
        report_by_model={name: full_report[name] for name in trained_model_names},
        model_selection_scores=model_selection_scores,
        save_path=utils.FIGURES_DIR / "models_compilation_overview.png",
    )
    utils.plot_model_status_overview(
        all_models_report=full_report,
        save_path=utils.FIGURES_DIR / "models_status_overview.png",
    )

    utils.plot_learning_curves(best_estimator, x_train_val, y_train_val, utils.FIGURES_DIR / "learning_curve_best_model.png")
    utils.plot_feature_importance_from_pipeline(best_estimator, utils.FIGURES_DIR / "feature_importance_best_model.png")

    model_path = _save_best_model_artifact(best_model_name, best_estimator)
    distilbert_note = _build_distilbert_note(distilbert_status, results)
    global_report = {
        "dataset_path": str(data_path),
        "n_samples": int(df.shape[0]),
        "best_model": best_model_name,
        "best_model_validation_metrics": best_model_payload["val_metrics"],
        "best_model_test_metrics": test_metrics[best_model_name],
        "best_model_cv_f1_macro_scores": best_cv_scores,
        "best_model_selection_score": model_selection_scores[best_model_name],
        "model_selection_method": {
            "formula": "selection_score = w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean",
            "weights": {"validation": selection_weights[0], "test": selection_weights[1], "cv": selection_weights[2]},
            "why_this_formula": (
                "Le F1 test reçoit le plus de poids pour refléter la performance hors entraînement; "
                "la validation sert de garde-fou pendant le tuning; la CV ajoute une mesure de stabilité."
            ),
            "tie_break_rule": "En cas d'égalité, choisir le meilleur f1_macro test puis accuracy test.",
            "cv_fallback_for_models": cv_fallback_models,
            "cv_fallback_explanation": (
                "Pour les modèles deep learning coûteux (ex. DistilBERT), le terme CV peut être approché "
                "par le score validation afin de garder un temps d'exécution raisonnable."
            ),
        },
        "model_selection_scores": model_selection_scores,
        "model_selection_ranking": ranking,
        "model_rationales": model_rationales,
        "best_model_selection_explanation": (
            f"Le modèle {best_model_name} obtient le meilleur score global pondéré "
            f"({model_selection_scores[best_model_name]:.4f}) en combinant validation, test et CV. "
            f"Il est retenu pour son compromis entre performance et robustesse."
        ),
        "distilbert_included": distilbert_enabled,
        "distilbert_note": distilbert_note,
        "expected_models": expected_model_names,
        "trained_models": trained_model_names,
        "model_execution_status": {name: full_report[name]["status"] for name in expected_model_names},
        "artifacts_generated": {
            "figures_dir": str(utils.FIGURES_DIR),
            "reports_dir": str(utils.REPORTS_DIR),
            "models_dir": str(utils.MODELS_DIR),
            "main_figures": [
                "models_compilation_overview.png",
                "models_status_overview.png",
                "confusion_matrices_all_models.png",
                "models_comparison_test.png",
                "learning_curve_best_model.png",
                "feature_importance_best_model.png",
            ],
        },
        "run_config": {
            "max_samples": max_samples,
            "distilbert_epochs": distilbert_epochs,
            "include_distilbert": include_distilbert,
            "test_size": test_size,
            "val_size": val_size,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "selection_weights": selection_weights,
            "random_state": random_state,
        },
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
    """Point d'entrée terminal: lance `run_pipeline` avec la configuration par défaut."""
    results = run_pipeline()
    print("Pipeline terminé.")
    print(f"Meilleur modèle: {results['best_model_name']}")
    print(f"Rapport JSON: {results['report_path']}")
    print(f"Figures: {results['figures_dir']}")


if __name__ == "__main__":
    main()
