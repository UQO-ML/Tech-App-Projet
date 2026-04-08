"""Helpers de workflow pour garder le notebook minimal."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from Code import utils
from Code.main import run_pipeline

DEFAULT_FIGURE_NAMES = [
    "runs_comparison_overview.png",
    "models_compilation_overview.png",
    "models_status_overview.png",
    "confusion_matrices_all_models.png",
    "models_comparison_test.png",
    "learning_curve_best_model.png",
]

def load_report(report_path: str | Path) -> dict[str, Any]:
    """Charge un report JSON de métriques.

    Paramètres:
        report_path: Chemin vers le report JSON.

    Retour:
        Dictionnaire du report.
    """
    path = Path(report_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_models_table(report: dict[str, Any]) -> pd.DataFrame:
    """Construit un tableau synthèse pour tous les modèles attendus.

    Paramètres:
        report: Dictionnaire `metrics_report.json`.

    Retour:
        DataFrame trié par statut puis score de sélection.
    """
    rows: list[dict[str, Any]] = []
    for model_name in report.get("expected_models", []):
        payload = report.get("all_models", {}).get(model_name, {})
        rows.append(
            {
                "model": model_name,
                "status": payload.get("status"),
                "error_or_reason": payload.get("error"),
                "selection_score": payload.get("selection_score"),
                "val_f1_macro": payload.get("validation_metrics", {}).get("f1_macro"),
                "test_f1_macro": payload.get("test_metrics", {}).get("f1_macro"),
                "cv_f1_macro_mean": payload.get("cv_f1_macro_mean"),
                "representation": payload.get("feature_config", {}).get("representation"),
                "best_cv_score": payload.get("tuning", {}).get("best_cv_score"),
                "best_params": str(payload.get("tuning", {}).get("best_params", {})),
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "selection_score"], ascending=[True, False])


def build_runs_comparison_table(reports_dir: str | Path, distilbert_proxy_penalty: float = 0.01) -> pd.DataFrame:
    """Construit la table de comparaison inter-runs depuis les reports.

    Paramètres:
        reports_dir: Dossier contenant `metrics_report_run_*.json`.
        distilbert_proxy_penalty: Malus appliqué quand un run inclut DistilBERT avec CV proxy.

    Retour:
        DataFrame trié par score global décroissant.
    """
    run_report_files = sorted(Path(reports_dir).glob("metrics_report_run_*.json"))
    rows: list[dict[str, Any]] = []
    for path in run_report_files:
        run_report = load_report(path)
        include_distilbert = bool(run_report.get("run_config", {}).get("include_distilbert", False))
        cv_fallback_models = run_report.get("model_selection_method", {}).get("cv_fallback_for_models", [])
        distilbert_cv_proxy = include_distilbert and ("DistilBERT" in cv_fallback_models)
        # Compensation réaliste: léger malus de prudence quand DistilBERT utilise un CV proxy.
        fairness_penalty = float(distilbert_proxy_penalty) if distilbert_cv_proxy else 0.0
        base_score = run_report.get("best_model_selection_score")
        adjusted_score = None if base_score is None else float(base_score - fairness_penalty)
        rows.append(
            {
                "run": path.stem.replace("metrics_report_", ""),
                "best_model": run_report.get("best_model"),
                "best_selection_score": base_score,
                "adjusted_selection_score": adjusted_score,
                "best_test_f1_macro": run_report.get("best_model_test_metrics", {}).get("f1_macro"),
                "include_distilbert": include_distilbert,
                "distilbert_cv_proxy": distilbert_cv_proxy,
                "fairness_penalty": fairness_penalty,
                "distilbert_note": run_report.get("distilbert_note"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("adjusted_selection_score", ascending=False)


def run_all_configs(runs: dict[str, dict[str, Any]], distilbert_proxy_penalty: float = 0.01) -> dict[str, Any]:
    """Exécute tous les runs, snapshot les artefacts, et active le meilleur run.

    Paramètres:
        runs: Dictionnaire des runs à exécuter (`why`, `config`).
        distilbert_proxy_penalty: Malus de prudence appliqué aux runs DistilBERT avec CV proxy.

    Retour:
        Dictionnaire contenant artefacts par run, run de référence et table de comparaison.
    """
    all_artifacts: dict[str, dict[str, Any]] = {}
    run_summaries: list[dict[str, Any]] = []
    runs_root = Path("Outputs") / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for run_name, run_ctx in runs.items():
        run_config = run_ctx["config"]
        print("=" * 90)
        print(f"Execution: {run_name}")
        print("Pourquoi:", run_ctx["why"])
        print("Configuration:", run_config)

        run_artifacts = run_pipeline(**run_config)
        all_artifacts[run_name] = run_artifacts

        outputs_dir = Path(run_artifacts["outputs_dir"])
        figures_src = Path(run_artifacts["figures_dir"])
        models_src = outputs_dir / "models"
        reports_src = outputs_dir / "reports"

        run_dir = runs_root / run_name
        run_figures_dir = run_dir / "figures"
        run_models_dir = run_dir / "models"
        if run_figures_dir.exists():
            shutil.rmtree(run_figures_dir)
        if run_models_dir.exists():
            shutil.rmtree(run_models_dir)
        shutil.copytree(figures_src, run_figures_dir)
        shutil.copytree(models_src, run_models_dir)

        report_src = Path(run_artifacts["report_path"])
        run_report_path = reports_src / f"metrics_report_{run_name}.json"
        shutil.copyfile(report_src, run_report_path)
        run_report = load_report(run_report_path)

        include_distilbert = bool(run_config.get("include_distilbert", False))
        cv_fallback_models = run_report.get("model_selection_method", {}).get("cv_fallback_for_models", [])
        distilbert_cv_proxy = include_distilbert and ("DistilBERT" in cv_fallback_models)
        fairness_penalty = float(distilbert_proxy_penalty) if distilbert_cv_proxy else 0.0
        base_score = run_report.get("best_model_selection_score")
        adjusted_score = None if base_score is None else float(base_score - fairness_penalty)

        run_summaries.append(
            {
                "run": run_name,
                "best_model": run_report.get("best_model"),
                "best_selection_score": base_score,
                "adjusted_selection_score": adjusted_score,
                "best_test_f1_macro": run_report.get("best_model_test_metrics", {}).get("f1_macro"),
                "include_distilbert": include_distilbert,
                "distilbert_cv_proxy": distilbert_cv_proxy,
                "fairness_penalty": fairness_penalty,
                "report_path": str(run_report_path),
            }
        )
        print(f"Report run sauvegarde: {run_report_path}")

    run_summary_df = pd.DataFrame(run_summaries).sort_values("adjusted_selection_score", ascending=False)
    best_run = str(run_summary_df.iloc[0]["run"])
    best_run_dir = runs_root / best_run
    best_artifacts = all_artifacts[best_run]

    outputs_dir = Path(best_artifacts["outputs_dir"])
    live_figures_dir = outputs_dir / "figures"
    live_models_dir = outputs_dir / "models"
    live_reports_dir = outputs_dir / "reports"
    if live_figures_dir.exists():
        shutil.rmtree(live_figures_dir)
    if live_models_dir.exists():
        shutil.rmtree(live_models_dir)
    shutil.copytree(best_run_dir / "figures", live_figures_dir)
    shutil.copytree(best_run_dir / "models", live_models_dir)

    best_report_path = live_reports_dir / f"metrics_report_{best_run}.json"
    shutil.copyfile(best_report_path, live_reports_dir / "metrics_report.json")
    utils.plot_runs_comparison(run_summary=run_summary_df, save_path=live_figures_dir / "runs_comparison_overview.png")

    return {
        "artifacts": best_artifacts,
        "best_run": best_run,
        "run_summary_df": run_summary_df,
        "all_artifacts": all_artifacts,
        "runs_root": runs_root,
        "figure_names": DEFAULT_FIGURE_NAMES,
        "distilbert_proxy_penalty": float(distilbert_proxy_penalty),
    }
