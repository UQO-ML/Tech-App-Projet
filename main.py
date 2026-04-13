#!/usr/bin/env python3
"""Point d'entrée CLI complet (équivalent orchestration notebook)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from notebook_workflow import load_report, run_all_configs
from result_interpreter import interpret_report
from run_configs import (
    DISTILBERT_PROXY_PENALTY_DEFAULT,
    filter_incompatible_runs,
    get_default_runs,
    get_exhaustive_runs,
)


def _parse_args() -> argparse.Namespace:
    """Parse les options CLI du lanceur."""
    parser = argparse.ArgumentParser(description="Exécute la pipeline multi-runs sans notebook.")
    parser.add_argument(
        "--run-matrix",
        choices=["default", "exhaustive"],
        default="exhaustive",
        help="Choix de matrice de runs (default plus rapide, exhaustive plus coûteuse).",
    )
    parser.add_argument(
        "--distilbert-proxy-penalty",
        type=float,
        default=DISTILBERT_PROXY_PENALTY_DEFAULT,
        help="Malus de prudence pour runs DistilBERT avec CV proxy (0.00 à 0.05 recommandé).",
    )
    return parser.parse_args()


def main() -> None:
    """Exécute l'orchestration complète des runs et affiche un résumé."""
    cli_start = time.perf_counter()
    args = _parse_args()

    t0 = time.perf_counter()
    runs = get_exhaustive_runs() if args.run_matrix == "exhaustive" else get_default_runs()
    runs, skipped_runs = filter_incompatible_runs(runs)
    print(f"Matrice de runs: {args.run_matrix} | nombre de runs: {len(runs)}")
    if skipped_runs:
        print("Runs ignorés (dépendances absentes dans l'environnement):", ", ".join(skipped_runs))
    print(f"[timing] build_and_validate_runs: {time.perf_counter() - t0:.2f}s")

    print(f"DISTILBERT_PROXY_PENALTY: {args.distilbert_proxy_penalty:.4f}")

    t0 = time.perf_counter()
    workflow = run_all_configs(runs, distilbert_proxy_penalty=args.distilbert_proxy_penalty)
    print(f"[timing] execute_all_runs: {time.perf_counter() - t0:.2f}s")

    best_run = workflow["best_run"]
    artifacts = workflow["artifacts"]
    run_summary_df = workflow["run_summary_df"]
    print("\n=== RESULTAT GLOBAL ===")
    print(f"Run de référence: {best_run}")
    print("Top 5 runs (score ajusté):")
    columns = [
        "run",
        "best_model",
        "adjusted_selection_score",
        "best_test_f1_macro",
        "distilbert_cv_proxy",
        "fairness_penalty",
    ]
    print(run_summary_df[columns].head(5).to_string(index=False))

    t0 = time.perf_counter()
    report = load_report(Path(artifacts["report_path"]))
    print("\n=== INTERPRETATION ===")
    interpret_report(report, weak_f1_threshold=0.55)
    print(f"[timing] interpret_best_run: {time.perf_counter() - t0:.2f}s")
    print("\nRapport:", artifacts["report_path"])
    print("Figures:", artifacts["figures_dir"])
    print(f"[timing] cli_total: {time.perf_counter() - cli_start:.2f}s")


if __name__ == "__main__":
    main()
