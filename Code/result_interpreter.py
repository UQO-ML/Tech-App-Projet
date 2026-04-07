"""Interpretation helpers for model selection reports."""

from __future__ import annotations

from typing import Any


def _fmt_metric(value: Any) -> str:
    """Format a metric value for terminal/notebook display.

    Parameters:
        value: Metric value that may be numeric or missing.

    Returns:
        A readable string (`n/a` if missing, else `X.XXXX`).
    """
    return "n/a" if value is None else f"{float(value):.4f}"


def _generalization_label(val_f1: float | None, test_f1: float | None) -> str:
    """Label generalization behavior from validation/test gap.

    Parameters:
        val_f1: Validation macro-F1.
        test_f1: Test macro-F1.

    Returns:
        Diagnostic label describing likely behavior.
    """
    if val_f1 is None or test_f1 is None:
        return "incomplet"
    gap = float(val_f1 - test_f1)
    if gap > 0.05:
        return "possible surapprentissage"
    if gap < -0.05:
        return "possible sous-ajustement / split favorable test"
    return "generalisation stable"


def _stability_label(cv_mean: float | None, test_f1: float | None) -> str:
    """Label stability based on test-vs-CV distance.

    Parameters:
        cv_mean: Mean cross-validation macro-F1.
        test_f1: Test macro-F1.

    Returns:
        Human-readable stability diagnosis.
    """
    if cv_mean is None or test_f1 is None:
        return "stabilite non disponible"
    delta = abs(float(test_f1 - cv_mean))
    if delta <= 0.03:
        return "stabilite forte"
    if delta <= 0.07:
        return "stabilite moyenne"
    return "stabilite faible"


def _cv_confidence_label(cv_ci95: float | None) -> str:
    """Qualifie l'incertitude CV selon la largeur de l'IC95."""
    if cv_ci95 is None:
        return "IC95 indisponible"
    if cv_ci95 <= 0.015:
        return "CV très stable"
    if cv_ci95 <= 0.035:
        return "CV stable"
    return "CV instable"


def interpret_report(report: dict[str, Any], weak_f1_threshold: float = 0.55) -> dict[str, Any]:
    """Print an interpretation summary and return structured insights.

    Parameters:
        report: Content of `metrics_report.json`.
        weak_f1_threshold: Threshold under which a trained model is flagged as weak.

    Returns:
        Dictionary with top models, weak models, and status counts.
    """
    all_models = report.get("all_models", {})
    ranking = report.get("model_selection_ranking", [])

    status_counts: dict[str, int] = {}
    for payload in all_models.values():
        status = payload.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    trained = [name for name, payload in all_models.items() if payload.get("status") == "trained"]
    top_models = ranking[:3]

    print("=== INTERPRETATION RAPIDE ===")
    print(f"Meilleur modele: {report.get('best_model')}")
    print(f"Score de selection: {_fmt_metric(report.get('best_model_selection_score'))}")
    print(
        "Politique scoring:",
        report.get("model_selection_method", {}).get("formula", "N/A"),
    )
    print(
        "Precision (diagnostic):",
        report.get("model_selection_method", {}).get("precision_policy", "N/A"),
    )
    print("\nStatuts des modeles:")
    for status, count in sorted(status_counts.items(), key=lambda item: item[0]):
        print(f"{status}: {count}")

    if trained:
        print("\nTop 3 modeles (score global):", top_models)

    print("\nDiagnostic par modele entraine:")
    for name in ranking:
        payload = all_models.get(name, {})
        if payload.get("status") != "trained":
            continue
        val_f1 = payload.get("validation_metrics", {}).get("f1_macro")
        test_f1 = payload.get("test_metrics", {}).get("f1_macro")
        cv_mean = payload.get("cv_f1_macro_mean")
        cv_ci95 = payload.get("cv_f1_macro_ci95")
        selection_score = payload.get("selection_score")
        balanced_acc = payload.get("test_metrics", {}).get("balanced_accuracy")
        cls_report = payload.get("classification_report_test", {})
        hate_f1 = cls_report.get("hate_speech", {}).get("f1-score")
        hate_recall = cls_report.get("hate_speech", {}).get("recall")
        penalty_applied = payload.get("selection_components", {}).get("penalty_applied")
        diagnosis = _generalization_label(val_f1, test_f1)
        stability = _stability_label(cv_mean, test_f1)
        cv_confidence = _cv_confidence_label(cv_ci95)
        print(
            f"- {name}: selection={_fmt_metric(selection_score)} | "
            f"val_f1={_fmt_metric(val_f1)} | test_f1={_fmt_metric(test_f1)} | "
            f"balanced_acc={_fmt_metric(balanced_acc)} | cv_mean={_fmt_metric(cv_mean)} "
            f"± {_fmt_metric(cv_ci95)} | hate_f1={_fmt_metric(hate_f1)} | "
            f"hate_recall={_fmt_metric(hate_recall)} | penalty={_fmt_metric(penalty_applied)} | "
            f"{diagnosis} | {stability} | {cv_confidence}"
        )

    weak_models: list[tuple[str, float]] = []
    for name in trained:
        f1 = all_models[name].get("test_metrics", {}).get("f1_macro")
        if f1 is not None and float(f1) < weak_f1_threshold:
            weak_models.append((name, float(f1)))
    weak_models.sort(key=lambda item: item[1])

    if weak_models:
        print(f"\nModeles a ameliorer (F1_macro test < {weak_f1_threshold:.2f}):")
        for name, f1 in weak_models:
            print(f"- {name}: {f1:.4f}")

    print("\nRecommandations automatiques:")
    if weak_models:
        print("- Augmenter la taille d'echantillon (MAX_SAMPLES=None) avant nouveau tuning.")
    print("- Tester CV_FOLDS=3 pour runs rapides, puis 5 pour consolidation finale.")
    print(
        "- Ajuster SELECTION_WEIGHTS selon l'objectif (test, robustesse CV, rappel hate_speech) "
        "et calibrer HATE_RECALL_FLOOR / HATE_RECALL_PENALTY."
    )
    print("- Contrôler prioritairement hate_recall_test et hate_f1_test avant toute sélection finale.")
    print("- Pour DistilBERT, augmenter DISTILBERT_EPOCHS progressivement (1 -> 2 -> 3).")
    print("\nDistilBERT:", report.get("distilbert_note"))
    print("=== FIN INTERPRETATION ===")

    return {
        "status_counts": status_counts,
        "top_models": top_models,
        "weak_models": weak_models,
    }
