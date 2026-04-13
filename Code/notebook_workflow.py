"""Helpers de workflow pour garder le notebook minimal."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import utils
from report_markdown import save_report_markdown, save_runs_comparison_markdown

DEFAULT_FIGURE_NAMES = [
    "runs_comparison_overview.png",
    "models_compilation_overview.png",
    "models_status_overview.png",
    "confusion_matrices_all_models.png",
    "models_comparison_test.png",
    "learning_curve_best_model.png",
    "feature_importance_best_model.png",
    "feature_importance_comparison_models.png",
]


def _distilbert_enabled_from_run_config(run_config: dict[str, Any]) -> bool:
    """Détermine si DistilBERT est réellement activé dans un run."""
    include_distilbert = bool(run_config.get("include_distilbert", False))
    switches = run_config.get("algorithm_switches", {})
    if isinstance(switches, dict) and "DistilBERT" in switches:
        return bool(switches.get("DistilBERT"))
    return include_distilbert


def _is_distilbert_cv_proxy_for_winner(run_report: dict[str, Any]) -> bool:
    """Vrai si le meilleur modèle est DistilBERT avec CV proxy."""
    best_model = run_report.get("best_model")
    cv_fallback_models = run_report.get("model_selection_method", {}).get("cv_fallback_for_models", [])
    return best_model == "DistilBERT" and ("DistilBERT" in cv_fallback_models)


def _run_pipeline_subprocess(run_name: str, run_config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Exécute `run_pipeline` dans un subprocess Python isolé.

    Paramètres:
        run_name: Identifiant du run (utilisé dans les logs).
        run_config: Dictionnaire de configuration du run.
        project_root: Racine du projet (cwd du subprocess).

    Retour:
        Dictionnaire d'artefacts retourné par `run_pipeline`.
    """
    runner_path = project_root / "Code" / "run_pipeline_subprocess.py"
    reports_dir = project_root / "Outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    result_path = reports_dir / f"subprocess_result_{run_name}.json"

    command = [
        sys.executable,
        str(runner_path),
        "--run-name",
        run_name,
        "--config-json",
        json.dumps(run_config),
        "--result-path",
        str(result_path),
    ]
    completed = subprocess.run(command, cwd=str(project_root), capture_output=True, text=True, check=False)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())

    if not result_path.exists():
        raise RuntimeError(f"Aucun résultat subprocess généré pour le run '{run_name}'.")

    payload = load_report(result_path)
    result_path.unlink(missing_ok=True)

    if completed.returncode != 0 or payload.get("status") != "ok":
        raise RuntimeError(
            f"Run '{run_name}' en échec via subprocess: "
            f"{payload.get('error', 'erreur inconnue')}"
        )

    return payload["artifacts"]

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
        test_cls_report = payload.get("classification_report_test", {})
        selection_components = payload.get("selection_components", {})
        rows.append(
            {
                "model": model_name,
                "status": payload.get("status"),
                "error_or_reason": payload.get("error"),
                "selection_score": payload.get("selection_score"),
                "balanced_accuracy": payload.get("test_metrics", {}).get("balanced_accuracy"),
                "val_f1_macro": payload.get("validation_metrics", {}).get("f1_macro"),
                "test_f1_macro": payload.get("test_metrics", {}).get("f1_macro"),
                "cv_f1_macro_mean": payload.get("cv_f1_macro_mean"),
                "cv_f1_macro_std": payload.get("cv_f1_macro_std"),
                "cv_f1_macro_ci95": payload.get("cv_f1_macro_ci95"),
                "hate_recall_test": selection_components.get("hate_recall_test"),
                "hate_f1_test": test_cls_report.get("hate_speech", {}).get("f1-score"),
                "offensive_f1_test": test_cls_report.get("offensive_language", {}).get("f1-score"),
                "neither_f1_test": test_cls_report.get("neither", {}).get("f1-score"),
                "penalty_applied": selection_components.get("penalty_applied"),
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
        include_distilbert = _distilbert_enabled_from_run_config(run_report.get("run_config", {}))
        distilbert_cv_proxy = _is_distilbert_cv_proxy_for_winner(run_report)
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
    failed_runs: list[dict[str, str]] = []
    runs_root = Path("Outputs") / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]

    for run_name, run_ctx in runs.items():
        run_config = run_ctx["config"]
        print("=" * 90)
        print(f"Execution: {run_name}")
        print("Pourquoi:", run_ctx["why"])
        print("Configuration:", run_config)

        try:
            run_artifacts = _run_pipeline_subprocess(run_name=run_name, run_config=run_config, project_root=project_root)
        except Exception as exc:
            error_message = str(exc)
            print(f"Run ignoré (échec): {run_name} -> {error_message}")
            failed_runs.append({"run": run_name, "error": error_message})
            continue
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
        save_report_markdown(run_report_path, reports_src / f"metrics_report_{run_name}.md")
        run_report = load_report(run_report_path)

        include_distilbert = _distilbert_enabled_from_run_config(run_config)
        distilbert_cv_proxy = _is_distilbert_cv_proxy_for_winner(run_report)
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

    if not run_summaries:
        raise RuntimeError(
            "Aucun run n'a abouti. "
            f"Runs en échec: {failed_runs}"
        )

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
    save_report_markdown(live_reports_dir / "metrics_report.json", live_reports_dir / "metrics_report.md")
    save_runs_comparison_markdown(run_summary_df, live_reports_dir / "runs_comparison_overview.md")
    utils.plot_runs_comparison(run_summary=run_summary_df, save_path=live_figures_dir / "runs_comparison_overview.png")

    return {
        "artifacts": best_artifacts,
        "best_run": best_run,
        "run_summary_df": run_summary_df,
        "all_artifacts": all_artifacts,
        "failed_runs": failed_runs,
        "runs_root": runs_root,
        "figure_names": DEFAULT_FIGURE_NAMES,
        "distilbert_proxy_penalty": float(distilbert_proxy_penalty),
    }
