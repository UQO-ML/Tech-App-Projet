#!/usr/bin/env python3
"""Point d'entrée CLI complet (équivalent orchestration notebook).

Ce script permet d'exécuter toute la pipeline multi-runs sans notebook:
1) construction d'une matrice de runs (incluant profils DistilBERT/MLP/AdaBoost),
2) exécution en subprocess de chaque run,
3) sélection automatique du meilleur run,
4) affichage d'un diagnostic texte.

Usage:
    python main.py
    python main.py --run-matrix default
    python main.py --run-matrix exhaustive --distilbert-proxy-penalty 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from notebook_workflow import load_report, run_all_configs
from result_interpreter import interpret_report

# Constantes partagées de configuration expérimentale.
TEST_SIZE = 0.2
VAL_SIZE = 0.1
CV_FOLDS_STANDARD = 5
SCORING = "f1_macro"
RANDOM_STATE = 42
MAX_SAMPLES_DEFAULT = None

SELECTION_WEIGHTS_BALANCED = (0.30, 0.35, 0.20, 0.15)
SELECTION_WEIGHTS_METHOD_STRICT = (0.45, 0.20, 0.20, 0.15)
SELECTION_WEIGHTS_CV_HEAVY = (0.25, 0.25, 0.35, 0.15)
SELECTION_WEIGHTS_DISTILBERT_SAFE = (0.30, 0.40, 0.15, 0.15)

HATE_RECALL_FLOOR = 0.40
HATE_RECALL_PENALTY = 0.03
DISTILBERT_PROXY_PENALTY_DEFAULT = 0.02

ALGORITHM_SWITCHES_ALL = {
    "NaiveBayes": True,
    "LogisticRegression": True,
    "LinearSVC": True,
    "KNN": True,
    "DecisionTree": True,
    "RandomForest": True,
    "AdaBoost": True,
    "MLPClassifier": True,
    "DistilBERT": True,
}
ALGORITHM_SWITCHES_CLASSIC_ONLY = {**ALGORITHM_SWITCHES_ALL, "DistilBERT": False}

MODEL_PARAM_OVERRIDES_BASE: dict[str, dict[str, Any]] = {}
MODEL_GRID_OVERRIDES_BASE: dict[str, dict[str, list[Any]]] = {}

DISTILBERT_PARAMS_EP3 = {
    "DistilBERT": {
        "epochs": 3,
        "batch_size": 16,
        "max_length": 160,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
    }
}
MLP_GRID_WIDE = {
    "MLPClassifier": {
        "svd__n_components": [20, 40, 60],
        "clf__hidden_layer_sizes": [(128,), (256, 128), (256, 256)],
        "clf__alpha": [1e-4, 5e-4, 1e-3],
        "clf__learning_rate_init": [1e-3, 5e-4, 3e-4],
    }
}


def get_default_runs() -> dict[str, dict[str, Any]]:
    """Construit une matrice de runs raisonnable (coût contrôlé)."""
    return {
        "run_a_data_balance": {
            "why": "Baseline robuste sur 100% des données.",
            "config": {
                "max_samples": MAX_SAMPLES_DEFAULT,
                "distilbert_epochs": 1,
                "include_distilbert": True,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "cv_folds": CV_FOLDS_STANDARD,
                "scoring": SCORING,
                "model_param_overrides": MODEL_PARAM_OVERRIDES_BASE,
                "model_grid_overrides": MODEL_GRID_OVERRIDES_BASE,
                "selection_weights": SELECTION_WEIGHTS_BALANCED,
                "algorithm_switches": ALGORITHM_SWITCHES_ALL,
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        },
        # "run_e_method_strict_classic": {
        #     "why": "Sélection méthodologique stricte (classiques).",
        #     "config": {
        #         "max_samples": MAX_SAMPLES_DEFAULT,
        #         "distilbert_epochs": 1,
        #         "include_distilbert": False,
        #         "test_size": TEST_SIZE,
        #         "val_size": VAL_SIZE,
        #         "cv_folds": CV_FOLDS_STANDARD,
        #         "scoring": SCORING,
        #         "model_param_overrides": MODEL_PARAM_OVERRIDES_BASE,
        #         "model_grid_overrides": MODEL_GRID_OVERRIDES_BASE,
        #         "selection_weights": SELECTION_WEIGHTS_METHOD_STRICT,
        #         "algorithm_switches": ALGORITHM_SWITCHES_CLASSIC_ONLY,
        #         "hate_recall_floor": HATE_RECALL_FLOOR,
        #         "hate_recall_penalty": HATE_RECALL_PENALTY,
        #         "random_state": RANDOM_STATE,
        #     },
        # },
        # "run_f_cv_heavy_classic": {
        #     "why": "Robustesse prioritaire via CV fort + grille MLP large.",
        #     "config": {
        #         "max_samples": MAX_SAMPLES_DEFAULT,
        #         "distilbert_epochs": 1,
        #         "include_distilbert": False,
        #         "test_size": TEST_SIZE,
        #         "val_size": VAL_SIZE,
        #         "cv_folds": CV_FOLDS_STANDARD,
        #         "scoring": SCORING,
        #         "model_param_overrides": MODEL_PARAM_OVERRIDES_BASE,
        #         "model_grid_overrides": MLP_GRID_WIDE,
        #         "selection_weights": SELECTION_WEIGHTS_CV_HEAVY,
        #         "algorithm_switches": ALGORITHM_SWITCHES_CLASSIC_ONLY,
        #         "hate_recall_floor": HATE_RECALL_FLOOR,
        #         "hate_recall_penalty": HATE_RECALL_PENALTY,
        #         "random_state": RANDOM_STATE,
        #     },
        # },
        "run_g_distilbert_safe_ep3": {
            "why": "DistilBERT ep3 avec compensation CV proxy.",
            "config": {
                "max_samples": MAX_SAMPLES_DEFAULT,
                "distilbert_epochs": 3,
                "include_distilbert": True,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "cv_folds": CV_FOLDS_STANDARD,
                "scoring": SCORING,
                "model_param_overrides": DISTILBERT_PARAMS_EP3,
                "model_grid_overrides": MODEL_GRID_OVERRIDES_BASE,
                "selection_weights": SELECTION_WEIGHTS_DISTILBERT_SAFE,
                "algorithm_switches": ALGORITHM_SWITCHES_ALL,
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        },
    }


def get_exhaustive_runs() -> dict[str, dict[str, Any]]:
    """Construit une matrice explicite (1 run = 1 config ciblée).

    Contrairement au produit cartésien, cette version crée des runs indépendants
    pour rechercher la meilleure configuration de chaque modèle séparément:
    - runs DistilBERT (DistilBERT seul actif),
    - runs MLP (MLP seul actif),
    - runs AdaBoost (AdaBoost seul actif).
    """
    base_runs = get_default_runs()
    exhaustive_runs: dict[str, dict[str, Any]] = {"run_a_data_balance": base_runs["run_a_data_balance"]}

    distilbert_profiles: dict[str, dict[str, dict[str, float | int]]] = {
        "fast": {"DistilBERT": {"epochs": 1, "batch_size": 16, "max_length": 128, "learning_rate": 5e-5, "weight_decay": 0.01}},
        "balanced": {"DistilBERT": {"epochs": 2, "batch_size": 16, "max_length": 160, "learning_rate": 3e-5, "weight_decay": 0.01}},
        "robust": {"DistilBERT": {"epochs": 3, "batch_size": 32, "max_length": 224, "learning_rate": 2e-5, "weight_decay": 0.02}},
        "vram_max": {"DistilBERT": {"epochs": 4, "batch_size": 48, "max_length": 320, "learning_rate": 2e-5, "weight_decay": 0.02}},
    }
    mlp_profiles: dict[str, dict[str, dict[str, list[Any]]]] = {
        "fast": {"MLPClassifier": {"svd__n_components": [20, 40], "clf__hidden_layer_sizes": [(128,)], "clf__alpha": [1e-4, 1e-3], "clf__learning_rate_init": [1e-3]}},
        "balanced": {"MLPClassifier": {"svd__n_components": [20, 40, 60], "clf__hidden_layer_sizes": [(128,), (256, 128)], "clf__alpha": [1e-4, 5e-4, 1e-3], "clf__learning_rate_init": [1e-3, 5e-4]}},
        "wide": {"MLPClassifier": {"svd__n_components": [20, 40, 60, 80], "clf__hidden_layer_sizes": [(128,), (256, 128), (256, 256)], "clf__alpha": [1e-5, 1e-4, 5e-4, 1e-3], "clf__learning_rate_init": [1e-3, 7e-4, 5e-4, 3e-4]}},
        "ultra": {"MLPClassifier": {"svd__n_components": [40, 80, 120], "clf__hidden_layer_sizes": [(256, 128), (256, 256), (512, 256)], "clf__alpha": [1e-6, 1e-5, 1e-4, 5e-4], "clf__learning_rate_init": [1e-3, 8e-4, 5e-4, 2e-4]}},
    }
    adaboost_profiles: dict[str, dict[str, dict[str, list[Any]]]] = {
        "fast": {"AdaBoost": {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.05, 0.1], "clf__estimator__max_depth": [1, 2], "clf__estimator__min_samples_leaf": [1, 2]}},
        "balanced": {"AdaBoost": {"clf__n_estimators": [200, 400, 800], "clf__learning_rate": [0.03, 0.1, 0.3], "clf__estimator__max_depth": [1, 2, 3], "clf__estimator__min_samples_leaf": [1, 2]}},
        "wide": {"AdaBoost": {"clf__n_estimators": [400, 800, 1200], "clf__learning_rate": [0.01, 0.03, 0.1, 0.3], "clf__estimator__max_depth": [1, 2, 3, 4], "clf__estimator__min_samples_leaf": [1, 2, 4]}},
        "ultra": {"AdaBoost": {"clf__n_estimators": [800, 1200, 1600], "clf__learning_rate": [0.01, 0.03, 0.1, 0.3, 0.6], "clf__estimator__max_depth": [2, 3, 4, 5], "clf__estimator__min_samples_leaf": [1, 2, 4, 8]}},
    }

    def _single_model_switches(target_model: str) -> dict[str, bool]:
        """Active uniquement le modèle ciblé dans un run."""
        return {name: (name == target_model) for name in ALGORITHM_SWITCHES_ALL}

    # Mapping explicite: 1 run = 1 profil DistilBERT.
    distilbert_run_map = {
        "run_d_fast": "fast",
        "run_d_balanced": "balanced",
        "run_d_robust": "robust",
        "run_d_vram_max": "vram_max",
    }
    for run_name, profile_name in distilbert_run_map.items():
        d_params = distilbert_profiles[profile_name]
        exhaustive_runs[run_name] = {
            "why": f"Run tuning DistilBERT seul (profil={profile_name}).",
            "config": {
                "max_samples": MAX_SAMPLES_DEFAULT,
                "distilbert_epochs": int(d_params["DistilBERT"]["epochs"]),
                "include_distilbert": True,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "cv_folds": CV_FOLDS_STANDARD,
                "scoring": SCORING,
                "model_param_overrides": d_params,
                "model_grid_overrides": MODEL_GRID_OVERRIDES_BASE,
                "selection_weights": SELECTION_WEIGHTS_DISTILBERT_SAFE,
                "algorithm_switches": _single_model_switches("DistilBERT"),
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        }

    # Mapping explicite: 1 run = 1 profil MLP.
    mlp_run_map = {
        "run_mlp_fast": "fast",
        "run_mlp_balanced": "balanced",
        "run_mlp_wide": "wide",
        "run_mlp_ultra": "ultra",
    }
    for run_name, profile_name in mlp_run_map.items():
        m_grid = mlp_profiles[profile_name]
        exhaustive_runs[run_name] = {
            "why": f"Run tuning MLP seul (profil={profile_name}).",
            "config": {
                "max_samples": MAX_SAMPLES_DEFAULT,
                "distilbert_epochs": 1,
                "include_distilbert": False,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "cv_folds": CV_FOLDS_STANDARD,
                "scoring": SCORING,
                "model_param_overrides": MODEL_PARAM_OVERRIDES_BASE,
                "model_grid_overrides": m_grid,
                "selection_weights": SELECTION_WEIGHTS_CV_HEAVY,
                "algorithm_switches": _single_model_switches("MLPClassifier"),
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        }

    # Mapping explicite: 1 run = 1 profil AdaBoost.
    adaboost_run_map = {
        "run_ada_fast": "fast",
        "run_ada_balanced": "balanced",
        "run_ada_wide": "wide",
        "run_ada_ultra": "ultra",
    }
    for run_name, profile_name in adaboost_run_map.items():
        a_grid = adaboost_profiles[profile_name]
        exhaustive_runs[run_name] = {
            "why": f"Run tuning AdaBoost seul (profil={profile_name}).",
            "config": {
                "max_samples": MAX_SAMPLES_DEFAULT,
                "distilbert_epochs": 1,
                "include_distilbert": False,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "cv_folds": CV_FOLDS_STANDARD,
                "scoring": SCORING,
                "model_param_overrides": MODEL_PARAM_OVERRIDES_BASE,
                "model_grid_overrides": a_grid,
                "selection_weights": SELECTION_WEIGHTS_TEST_PRIORITY,
                "algorithm_switches": _single_model_switches("AdaBoost"),
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        }
    return exhaustive_runs


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
    args = _parse_args()
    runs = get_exhaustive_runs() if args.run_matrix == "exhaustive" else get_default_runs()
    print(f"Matrice de runs: {args.run_matrix} | nombre de runs: {len(runs)}")
    print(f"DISTILBERT_PROXY_PENALTY: {args.distilbert_proxy_penalty:.4f}")
    workflow = run_all_configs(runs, distilbert_proxy_penalty=args.distilbert_proxy_penalty)

    best_run = workflow["best_run"]
    artifacts = workflow["artifacts"]
    run_summary_df = workflow["run_summary_df"]
    print("\n=== RESULTAT GLOBAL ===")
    print(f"Run de référence: {best_run}")
    print("Top 5 runs (score ajusté):")
    columns = ["run", "best_model", "adjusted_selection_score", "best_test_f1_macro", "distilbert_cv_proxy", "fairness_penalty"]
    print(run_summary_df[columns].head(5).to_string(index=False))

    report = load_report(Path(artifacts["report_path"]))
    print("\n=== INTERPRETATION ===")
    interpret_report(report, weak_f1_threshold=0.55)
    print("\nRapport:", artifacts["report_path"])
    print("Figures:", artifacts["figures_dir"])


if __name__ == "__main__":
    main()
