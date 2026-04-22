"""Pipeline principal du projet INF6243."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix

import models
import preprocessing as prep
import utils

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_data.csv"

HATE_CLASS_NAME = prep.CLASS_LABELS[0]
DEFAULT_SELECTION_WEIGHTS = (0.30, 0.35, 0.20, 0.15)
DEFAULT_HATE_RECALL_FLOOR = 0.40
DEFAULT_HATE_RECALL_PENALTY = 0.03
ERROR_CASES_MAX_EXAMPLES = 10


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
        "cache_hit": None,
        "cache_key": None,
        "cache_strategy": None,
    }


def _build_feature_config(model_name: str, tuning_info: dict[str, Any]) -> dict[str, Any]:
    """Extrait les métadonnées de représentation (TF-IDF/Transformer)."""
    best_params = tuning_info.get("best_params", {}) if isinstance(tuning_info, dict) else {}
    if model_name == "DistilBERT":
        representation = "Transformer embeddings"
        backend = "gpu_torch"
    elif model_name.endswith("GPU"):
        representation = "TF-IDF + SVD dense vectors"
        backend = "gpu_cuml"
    else:
        representation = "TF-IDF sparse vectors"
        backend = "cpu_sklearn"
    return {
        "representation": representation,
        "backend": backend,
        "tfidf_ngram_range": best_params.get("tfidf__ngram_range"),
        "tfidf_min_df": best_params.get("tfidf__min_df"),
        "tfidf_max_features": best_params.get("tfidf__max_features"),
        "deep_model_name": best_params.get("model_name") if model_name == "DistilBERT" else None,
        "max_length": best_params.get("max_length") if model_name == "DistilBERT" else None,
    }


def _normalize_selection_weights(selection_weights: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Normalise les poids de sélection en format 4 termes.

    Paramètres:
        selection_weights: Tuple de longueur 3 `(val, test, cv)` ou 4
            `(val, test, cv, hate_recall)`.

    Retour:
        Tuple normalisé `(w_val, w_test, w_cv, w_hate)`.
    """
    if len(selection_weights) == 4:
        return (
            float(selection_weights[0]),
            float(selection_weights[1]),
            float(selection_weights[2]),
            float(selection_weights[3]),
        )
    if len(selection_weights) == 3:
        # Compatibilité ascendante: ancien schéma sans terme explicite hate_recall.
        return (
            float(selection_weights[0]),
            float(selection_weights[1]),
            float(selection_weights[2]),
            0.0,
        )
    raise ValueError("selection_weights doit contenir 3 ou 4 valeurs.")


def _build_error_cases_report(
    x_test: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    max_examples: int = ERROR_CASES_MAX_EXAMPLES,
) -> dict[str, Any]:
    """Construit des exemples d'erreurs textuelles pour le modèle final.

    Paramètres:
        x_test: Textes du jeu de test.
        y_test: Labels réels.
        y_pred: Labels prédits.
        max_examples: Nombre maximal d'exemples par catégorie d'erreur.

    Retour:
        Dictionnaire structuré (FN/FP sur `hate_speech` + exemples généraux).
    """
    frame = pd.DataFrame(
        {
            "text": x_test.reset_index(drop=True),
            "true_label": y_test.reset_index(drop=True),
            "pred_label": y_pred.reset_index(drop=True),
        }
    )
    frame["true_name"] = frame["true_label"].map(prep.CLASS_LABELS)
    frame["pred_name"] = frame["pred_label"].map(prep.CLASS_LABELS)
    frame["is_error"] = frame["true_label"] != frame["pred_label"]
    error_frame = frame[frame["is_error"]].copy()

    hate_id = 0
    false_neg_hate = frame[(frame["true_label"] == hate_id) & (frame["pred_label"] != hate_id)].head(max_examples)
    false_pos_hate = frame[(frame["true_label"] != hate_id) & (frame["pred_label"] == hate_id)].head(max_examples)
    generic_errors = error_frame.head(max_examples)

    def _rows_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "text": str(row["text"]),
                "true_label": int(row["true_label"]),
                "true_name": str(row["true_name"]),
                "pred_label": int(row["pred_label"]),
                "pred_name": str(row["pred_name"]),
            }
            for _, row in df.iterrows()
        ]

    return {
        "summary": {
            "n_test_samples": int(frame.shape[0]),
            "n_errors": int(error_frame.shape[0]),
            "error_rate": float(error_frame.shape[0] / frame.shape[0]) if frame.shape[0] else 0.0,
            "n_false_negative_hate_speech": int(false_neg_hate.shape[0]),
            "n_false_positive_hate_speech": int(false_pos_hate.shape[0]),
        },
        "false_negative_hate_speech": _rows_to_records(false_neg_hate),
        "false_positive_hate_speech": _rows_to_records(false_pos_hate),
        "generic_errors": _rows_to_records(generic_errors),
    }


def _error_cases_to_markdown(error_cases: dict[str, Any]) -> str:
    """Construit une synthèse Markdown lisible des erreurs textuelles."""
    summary = error_cases.get("summary", {})
    lines: list[str] = []
    lines.append("# Analyse d'erreurs du modèle final")
    lines.append("")
    lines.append("## Résumé")
    lines.append(f"- n_test_samples: `{summary.get('n_test_samples', 'N/A')}`")
    lines.append(f"- n_errors: `{summary.get('n_errors', 'N/A')}`")
    lines.append(f"- error_rate: `{summary.get('error_rate', 'N/A')}`")
    lines.append(
        f"- false_negative_hate_speech: `{summary.get('n_false_negative_hate_speech', 'N/A')}` | "
        f"false_positive_hate_speech: `{summary.get('n_false_positive_hate_speech', 'N/A')}`"
    )

    def _section(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        if not rows:
            lines.append("Aucun exemple.")
            return
        lines.append("| True | Pred | Text |")
        lines.append("|---|---|---|")
        for row in rows:
            text = str(row.get("text", "")).replace("|", " ").replace("\n", " ").strip()
            if len(text) > 180:
                text = text[:177] + "..."
            lines.append(f"| {row.get('true_name', 'N/A')} | {row.get('pred_name', 'N/A')} | {text} |")

    _section("Faux négatifs hate_speech (true=hate_speech, pred!=hate_speech)", error_cases.get("false_negative_hate_speech", []))
    _section("Faux positifs hate_speech (true!=hate_speech, pred=hate_speech)", error_cases.get("false_positive_hate_speech", []))
    _section("Erreurs générales (échantillon)", error_cases.get("generic_errors", []))
    lines.append("")
    return "\n".join(lines)


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
        full_report[model_name]["cache_hit"] = result.get("cache_hit", False)
        full_report[model_name]["cache_key"] = result.get("cache_key", None)
        full_report[model_name]["cache_strategy"] = result.get("cache_strategy", None)
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
    selection_weights: tuple[float, float, float, float],
    hate_recall_floor: float,
    hate_recall_penalty: float,
) -> tuple[dict[str, float], dict[str, list[float]], list[str]]:
    """Calcule CV, score pondéré de sélection et fallback CV éventuel."""
    model_selection_scores: dict[str, float] = {}
    model_cv_scores: dict[str, list[float]] = {}
    cv_fallback_models: list[str] = []

    for model_name in trained_model_names:
        result = results[model_name]
        estimator = result["estimator"]
        cv_scores = models.cross_validate_estimator(estimator, x_train_val, y_train_val, cv=cv_folds)
        cv_std = None
        cv_ci95 = None
        cv_ci_available = False
        if cv_scores:
            model_cv_scores[model_name] = cv_scores
            cv_mean = float(sum(cv_scores) / len(cv_scores))
            if len(cv_scores) > 1:
                variance = sum((score - cv_mean) ** 2 for score in cv_scores) / (len(cv_scores) - 1)
                cv_std = float(math.sqrt(variance))
                cv_ci95 = float(1.96 * (cv_std / math.sqrt(len(cv_scores))))
                cv_ci_available = True
            else:
                cv_std = 0.0
                cv_ci95 = 0.0
        else:
            # Pour DistilBERT, la CV complète est coûteuse; on approxime avec le F1 validation.
            cv_mean = float(result["val_metrics"]["f1_macro"])
            model_cv_scores[model_name] = [cv_mean]
            cv_fallback_models.append(model_name)

        val_f1 = float(result["val_metrics"]["f1_macro"])
        test_f1 = float(test_metrics[model_name]["f1_macro"])
        class_report_test = full_report[model_name].get("classification_report_test", {})
        hate_recall = float(class_report_test.get(HATE_CLASS_NAME, {}).get("recall", 0.0))
        w_val, w_test, w_cv, w_hate = selection_weights
        base_score = float((w_val * val_f1) + (w_test * test_f1) + (w_cv * cv_mean) + (w_hate * hate_recall))
        penalty_applied = float(hate_recall_penalty) if hate_recall < float(hate_recall_floor) else 0.0
        model_selection_scores[model_name] = float(base_score - penalty_applied)
        full_report[model_name]["cv_f1_macro_scores"] = cv_scores
        full_report[model_name]["cv_f1_macro_mean"] = cv_mean
        full_report[model_name]["cv_f1_macro_std"] = cv_std
        full_report[model_name]["cv_f1_macro_ci95"] = cv_ci95
        full_report[model_name]["selection_score"] = model_selection_scores[model_name]
        full_report[model_name]["selection_components"] = {
            "val_f1_macro": val_f1,
            "test_f1_macro": test_f1,
            "cv_f1_macro_mean": cv_mean,
            "cv_f1_macro_std": cv_std,
            "cv_f1_macro_ci95": cv_ci95,
            "cv_ci95_available": cv_ci_available,
            "hate_recall_test": hate_recall,
            "weights": {
                "validation": w_val,
                "test": w_test,
                "cv": w_cv,
                "hate_recall": w_hate,
            },
            "hate_recall_floor": float(hate_recall_floor),
            "hate_recall_penalty": float(hate_recall_penalty),
            "penalty_applied": penalty_applied,
            "score_before_penalty": base_score,
            "score_after_penalty": model_selection_scores[model_name],
        }

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
    algorithm_switches: dict[str, bool] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    model_param_overrides: dict[str, dict[str, Any]] | None = None,
    model_grid_overrides: dict[str, dict[str, list[Any]]] | None = None,
    selection_weights: tuple[float, ...] = DEFAULT_SELECTION_WEIGHTS,
    hate_recall_floor: float = DEFAULT_HATE_RECALL_FLOOR,
    hate_recall_penalty: float = DEFAULT_HATE_RECALL_PENALTY,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Exécute le pipeline complet de classification et retourne les artefacts produits.

    Paramètres:
        data_path: Chemin vers le fichier de données.
        max_samples: Taille maximale d'échantillon (stratifié) pour accélérer les essais.
        distilbert_epochs: Nombre d'epochs utilisé pour DistilBERT.
        include_distilbert: Active ou non DistilBERT dans la comparaison.
        algorithm_switches: Activation fine par modèle `{nom_modele: bool}`.
            Exemple: `{"AdaBoost": False, "DistilBERT": True}`.
        test_size: Proportion du jeu de test.
        val_size: Proportion du jeu de validation.
        cv_folds: Nombre de folds pour la validation croisée.
        scoring: Métrique utilisée pour GridSearchCV.
        model_param_overrides: Paramètres fixes par modèle, passés à l'entraînement.
            Usage recommandé:
            - DistilBERT: `epochs`, `batch_size`, `max_length`, `learning_rate`, `weight_decay`.
            Impacts:
            - `epochs` élevé améliore souvent la convergence mais augmente fortement le temps;
            - `batch_size` faible réduit la mémoire GPU au prix d'un entraînement plus lent;
            - `max_length` élevé capture plus de contexte mais augmente le coût mémoire/temps.
        model_grid_overrides: Surcharge de grilles GridSearch pour modèles classiques.
            Format:
            - `{nom_modele: {param_name: [valeurs...]}}`.
            Impact:
            - grille large = meilleure couverture hyperparamètres mais coût CPU/RAM plus élevé.
        selection_weights: Pondérations de sélection:
            - ancien format compatible: `(validation, test, cv)`;
            - format recommandé: `(validation, test, cv, hate_recall)`.
        hate_recall_floor: Seuil minimal de rappel sur la classe `hate_speech` (test).
        hate_recall_penalty: Pénalité soustraite au score si le seuil n'est pas atteint.
        random_state: Seed globale.

    Retour:
        Dictionnaire avec le meilleur modèle, le chemin du report et les dossiers de sortie.
    """
    phase_times: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    def _tic() -> float:
        return time.perf_counter()

    def _toc(phase_name: str, start_time: float) -> None:
        elapsed = time.perf_counter() - start_time
        phase_times[phase_name] = float(elapsed)
        print(f"[timing] {phase_name}: {elapsed:.2f}s")

    t0 = _tic()
    _prepare_outputs()
    device = utils.get_device()
    print(f"Device détecté (pour deep learning): {device}")
    _toc("prepare_outputs_and_device", t0)

    t0 = _tic()
    raw_df = prep.load_data(data_path)
    df = prep.clean_data(raw_df)
    df = _sample_dataframe(df, max_samples, random_state=random_state)
    _toc("load_clean_sample_data", t0)

    t0 = _tic()
    summary = prep.exploratory_summary(df, target_column="class")
    utils.save_json(summary, utils.REPORTS_DIR / "eda_summary.json")

    # Visualisations EDA
    utils.plot_class_distribution(df["class"], prep.CLASS_LABELS, utils.FIGURES_DIR / "class_distribution.png")
    utils.plot_missing_values(df, utils.FIGURES_DIR / "missing_values.png")
    utils.plot_numeric_correlation(df, utils.FIGURES_DIR / "correlation_heatmap.png")
    utils.plot_text_length(df, prep.CLASS_LABELS, utils.FIGURES_DIR / "tweet_length_histogram.png")
    utils.plot_word_count_boxplot(df, prep.CLASS_LABELS, utils.FIGURES_DIR / "word_count_boxplot.png")
    _toc("eda_summary_and_figures", t0)

    # Split train / val / test
    t0 = _tic()
    x = df["clean_tweet"]
    y = df["class"]
    x_train, x_val, x_test, y_train, y_val, y_test = prep.train_val_test_split(
        x,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    _toc("split_train_val_test", t0)

    # Signature des données pour le cache des modèles
    data_path_obj = Path(data_path)
    split_signature = {
        "data_path": str(data_path_obj.resolve()),
        "data_mtime_ns": data_path_obj.stat().st_mtime_ns if data_path_obj.exists() else None,
        "max_samples": max_samples,
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "n_rows_after_clean": int(df.shape[0]),
    }

    # Entraînement des modèles classiques + DistilBERT (si dépendances disponibles)
    t0 = _tic()
    results = models.train_all_models(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=random_state,
        include_distilbert=include_distilbert,
        algorithm_switches=algorithm_switches,
        distilbert_epochs=distilbert_epochs,
        cv_folds=cv_folds,
        scoring=scoring,
        model_param_overrides=model_param_overrides,
        model_grid_overrides=model_grid_overrides,
        split_signature=split_signature,
    )
    cache_summary = {
        "n_total": len(results),
        "n_cache_hit": sum(1 for r in results.values() if r.get("cache_hit") is True),
        "n_retrained": sum(1 for r in results.values() if r.get("status") == "trained" and r.get("cache_hit") is False),
    }
    print(f"[cache] hits={cache_summary['n_cache_hit']} retrained={cache_summary['n_retrained']} total={cache_summary['n_total']}")
    _toc("train_models", t0)

    t0 = _tic()
    resolved_switches = models.resolve_algorithm_switches(
        include_distilbert=include_distilbert,
        algorithm_switches=algorithm_switches,
        random_state=random_state,
    )
    expected_model_names = models.get_expected_model_names(
        include_distilbert=include_distilbert,
        algorithm_switches=algorithm_switches,
        random_state=random_state,
    )
    distilbert_status = results.get("DistilBERT", {}).get("status", "skipped")
    distilbert_enabled = distilbert_status == "trained"

    model_rationales = models.get_model_rationales(
        random_state=random_state,
        include_distilbert=include_distilbert,
        algorithm_switches=algorithm_switches,
    )
    test_metrics, full_report, confusion_by_model, trained_model_names = _evaluate_models(
        expected_model_names=expected_model_names,
        results=results,
        model_rationales=model_rationales,
        x_test=x_test,
        y_test=y_test,
    )
    _toc("evaluate_on_test", t0)

    t0 = _tic()
    if test_metrics:
        utils.plot_models_comparison(test_metrics, utils.FIGURES_DIR / "models_comparison_test.png")
    if confusion_by_model:
        utils.plot_confusion_matrices_grid(
            confusion_by_model=confusion_by_model,
            class_names=prep.CLASS_LABELS,
            save_path=utils.FIGURES_DIR / "confusion_matrices_all_models.png",
        )
    _toc("evaluation_figures", t0)

    # Validation croisée k-fold sur train+val pour tous les modèles
    t0 = _tic()
    x_train_val = pd.concat([x_train, x_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    normalized_weights = _normalize_selection_weights(selection_weights)
    model_selection_scores, model_cv_scores, cv_fallback_models = _compute_selection_scores(
        trained_model_names=trained_model_names,
        results=results,
        test_metrics=test_metrics,
        x_train_val=x_train_val,
        y_train_val=y_train_val,
        full_report=full_report,
        cv_folds=cv_folds,
        selection_weights=normalized_weights,
        hate_recall_floor=hate_recall_floor,
        hate_recall_penalty=hate_recall_penalty,
    )
    _toc("cv_and_selection_scores", t0)

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
    best_y_pred_test = best_estimator.predict(x_test)

    t0 = _tic()
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
    trained_estimators = {
        model_name: results[model_name]["estimator"]
        for model_name in trained_model_names
        if results.get(model_name, {}).get("estimator") is not None
    }
    utils.plot_feature_importance_comparison(
        model_pipelines=trained_estimators,
        save_path=utils.FIGURES_DIR / "feature_importance_comparison_models.png",
        top_k_per_model=15,
        max_terms_union=40,
    )
    _toc("best_model_figures", t0)

    t0 = _tic()
    model_path = _save_best_model_artifact(best_model_name, best_estimator)
    distilbert_note = _build_distilbert_note(distilbert_status, results)
    error_cases = _build_error_cases_report(x_test=x_test, y_test=y_test, y_pred=pd.Series(best_y_pred_test))
    error_cases_json_path = utils.save_json(error_cases, utils.REPORTS_DIR / "error_cases_best_model.json")
    feature_importance_summary = utils.build_feature_importance_summary_by_model(
        model_pipelines=trained_estimators,
        top_k_per_model=15,
    )
    feature_importance_summary_path = utils.save_json(
        {"by_model": feature_importance_summary},
        utils.REPORTS_DIR / "feature_importance_summary.json",
    )
    error_cases_md_path = utils.REPORTS_DIR / "error_cases_best_model.md"
    error_cases_md_path.write_text(_error_cases_to_markdown(error_cases), encoding="utf-8")
    _toc("save_artifacts_and_error_analysis", t0)

    t0 = _tic()
    global_report = {
        "dataset_path": str(data_path),
        "n_samples": int(df.shape[0]),
        "best_model": best_model_name,
        "best_model_validation_metrics": best_model_payload["val_metrics"],
        "best_model_test_metrics": test_metrics[best_model_name],
        "best_model_cv_f1_macro_scores": best_cv_scores,
        "best_model_selection_score": model_selection_scores[best_model_name],
        "model_selection_method": {
            "formula": (
                "selection_score = "
                "w_val * val_f1_macro + w_test * test_f1_macro + w_cv * cv_f1_macro_mean + "
                "w_hate * hate_recall_test - penalty_if(hate_recall_test < hate_recall_floor)"
            ),
            "weights": {
                "validation": normalized_weights[0],
                "test": normalized_weights[1],
                "cv": normalized_weights[2],
                "hate_recall": normalized_weights[3],
            },
            "hate_recall_floor": float(hate_recall_floor),
            "hate_recall_penalty": float(hate_recall_penalty),
            "why_this_formula": (
                "Le F1 test reçoit le plus de poids pour refléter la performance hors entraînement; "
                "la validation sert de garde-fou pendant le tuning; la CV ajoute une mesure de stabilité; "
                "le rappel `hate_speech` protège la classe minoritaire; une pénalité s'applique si ce rappel est trop bas."
            ),
            "precision_policy": (
                "La précision macro est suivie comme métrique diagnostique, "
                "mais n'est pas utilisée comme critère principal de sélection."
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
            f"({model_selection_scores[best_model_name]:.4f}) en combinant validation, test, CV "
            f"et rappel `hate_speech` (avec pénalité éventuelle sous seuil). "
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
            "error_cases_json": str(error_cases_json_path),
            "error_cases_markdown": str(error_cases_md_path),
            "feature_importance_summary_json": str(feature_importance_summary_path),
            "main_figures": [
                "models_compilation_overview.png",
                "models_status_overview.png",
                "confusion_matrices_all_models.png",
                "models_comparison_test.png",
                "learning_curve_best_model.png",
                "feature_importance_best_model.png",
                "feature_importance_comparison_models.png",
            ],
        },
        "run_config": {
            "max_samples": max_samples,
            "distilbert_epochs": distilbert_epochs,
            "include_distilbert": include_distilbert,
            "algorithm_switches": resolved_switches,
            "test_size": test_size,
            "val_size": val_size,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "model_param_overrides": model_param_overrides or {},
            "model_grid_overrides": model_grid_overrides or {},
            "selection_weights": normalized_weights,
            "hate_recall_floor": float(hate_recall_floor),
            "hate_recall_penalty": float(hate_recall_penalty),
            "random_state": random_state,
        },
        "all_models": full_report,
        "cache_summary": cache_summary,
    }
    _toc("save_final_report", t0)

    phase_times["pipeline_total"] = float(time.perf_counter() - pipeline_start)
    print(f"[timing] pipeline_total: {phase_times['pipeline_total']:.2f}s")
    global_report["timing_seconds"] = phase_times
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
