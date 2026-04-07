"""Génération de rapports Markdown lisibles depuis les artefacts JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, ndigits: int = 4) -> str:
    """Formate une valeur scalaire pour affichage Markdown."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def _status_counts(report: dict[str, Any]) -> dict[str, int]:
    """Compte les statuts de modèles dans `all_models`."""
    counts = {"trained": 0, "skipped": 0, "failed": 0}
    for payload in report.get("all_models", {}).values():
        status = payload.get("status", "skipped")
        if status in counts:
            counts[status] += 1
        else:
            counts["skipped"] += 1
    return counts


def report_to_markdown(report: dict[str, Any]) -> str:
    """Construit une version Markdown lisible de `metrics_report.json`."""
    best_model = report.get("best_model", "N/A")
    best_score = _fmt(report.get("best_model_selection_score"))
    best_test_f1 = _fmt(report.get("best_model_test_metrics", {}).get("f1_macro"))
    cfg = report.get("run_config", {})
    method = report.get("model_selection_method", {})
    counts = _status_counts(report)

    lines: list[str] = []
    lines.append("# Rapport de métriques (lisible)")
    lines.append("")
    lines.append("## Résumé global")
    lines.append(f"- Modèle retenu: `{best_model}`")
    lines.append(f"- Score de sélection: `{best_score}`")
    lines.append(f"- F1 macro test du meilleur: `{best_test_f1}`")
    lines.append(f"- Échantillons: `{report.get('n_samples', 'N/A')}`")
    lines.append("")
    lines.append("## Statuts d'exécution")
    lines.append(f"- trained: `{counts['trained']}`")
    lines.append(f"- skipped: `{counts['skipped']}`")
    lines.append(f"- failed: `{counts['failed']}`")
    lines.append(f"- modèles attendus: `{report.get('expected_models', [])}`")
    lines.append(f"- modèles entraînés: `{report.get('trained_models', [])}`")
    lines.append("")
    lines.append("## Méthode de sélection")
    lines.append(f"- Formule: `{method.get('formula', 'N/A')}`")
    lines.append(
        f"- Poids: validation={_fmt(method.get('weights', {}).get('validation'))}, "
        f"test={_fmt(method.get('weights', {}).get('test'))}, "
        f"cv={_fmt(method.get('weights', {}).get('cv'))}, "
        f"hate_recall={_fmt(method.get('weights', {}).get('hate_recall'))}"
    )
    lines.append(f"- Seuil hate_recall: `{_fmt(method.get('hate_recall_floor'))}`")
    lines.append(f"- Pénalité hate_recall: `{_fmt(method.get('hate_recall_penalty'))}`")
    lines.append(f"- Politique précision: `{method.get('precision_policy', 'N/A')}`")
    lines.append(f"- Modèles avec CV proxy: `{method.get('cv_fallback_for_models', [])}`")
    lines.append("")
    lines.append("## Configuration du run")
    for key in (
        "max_samples",
        "distilbert_epochs",
        "include_distilbert",
        "algorithm_switches",
        "test_size",
        "val_size",
        "cv_folds",
        "scoring",
        "model_param_overrides",
        "model_grid_overrides",
        "selection_weights",
        "hate_recall_floor",
        "hate_recall_penalty",
        "random_state",
    ):
        lines.append(f"- {key}: `{cfg.get(key, 'N/A')}`")
    lines.append("")
    lines.append("## Détail par modèle")
    lines.append("")
    lines.append(
        "| Modèle | Status | Selection score | Balanced Acc | Val F1 | Test F1 | "
        "CV mean ± CI95 | Hate recall | Hate F1 | Pénalité appliquée | Erreur |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for model_name in report.get("expected_models", []):
        payload = report.get("all_models", {}).get(model_name, {})
        selection_components = payload.get("selection_components", {})
        cls_report = payload.get("classification_report_test", {})
        cv_mean = payload.get("cv_f1_macro_mean")
        cv_ci95 = payload.get("cv_f1_macro_ci95")
        if cv_mean is None:
            cv_mean_ci = "n/a"
        elif cv_ci95 is None:
            cv_mean_ci = f"{_fmt(cv_mean)} ± n/a"
        else:
            cv_mean_ci = f"{_fmt(cv_mean)} ± {_fmt(cv_ci95)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    str(payload.get("status", "N/A")),
                    _fmt(payload.get("selection_score")),
                    _fmt(payload.get("test_metrics", {}).get("balanced_accuracy")),
                    _fmt(payload.get("validation_metrics", {}).get("f1_macro")),
                    _fmt(payload.get("test_metrics", {}).get("f1_macro")),
                    cv_mean_ci,
                    _fmt(selection_components.get("hate_recall_test")),
                    _fmt(cls_report.get("hate_speech", {}).get("f1-score")),
                    _fmt(selection_components.get("penalty_applied")),
                    str(payload.get("error", "") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Analyse d'erreurs textuelles")
    lines.append("- Fichier JSON: `Outputs/reports/error_cases_best_model.json`")
    lines.append("- Fichier Markdown: `Outputs/reports/error_cases_best_model.md`")
    lines.append("- Résumé features par modèle: `Outputs/reports/feature_importance_summary.json`")
    lines.append("- Heatmap comparative: `Outputs/figures/feature_importance_comparison_models.png`")
    lines.append("")
    return "\n".join(lines)


def save_report_markdown(report_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Génère un `.md` lisible à partir d'un `metrics_report*.json`."""
    report_file = Path(report_path)
    with report_file.open("r", encoding="utf-8") as file:
        report = json.load(file)
    markdown = report_to_markdown(report)
    md_path = Path(output_path) if output_path is not None else report_file.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as file:
        file.write(markdown)
    return md_path


def runs_comparison_to_markdown(run_summary_df: pd.DataFrame) -> str:
    """Construit un résumé Markdown de la comparaison inter-runs."""
    if run_summary_df.empty:
        return "# Comparaison inter-runs\n\nAucun run disponible."

    frame = run_summary_df.copy()
    lines: list[str] = []
    lines.append("# Comparaison inter-runs")
    lines.append("")
    lines.append("| Run | Best model | Selection | Adjusted | Test F1 | DistilBERT CV proxy | Penalty |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run", "N/A")),
                    str(row.get("best_model", "N/A")),
                    _fmt(row.get("best_selection_score")),
                    _fmt(row.get("adjusted_selection_score")),
                    _fmt(row.get("best_test_f1_macro")),
                    str(row.get("distilbert_cv_proxy", False)),
                    _fmt(row.get("fairness_penalty")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def save_runs_comparison_markdown(run_summary_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Écrit un fichier Markdown pour la comparaison inter-runs."""
    md_path = Path(output_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as file:
        file.write(runs_comparison_to_markdown(run_summary_df))
    return md_path
