"""Configurations de runs partagées entre notebook et CLI.

Ce module centralise la définition des matrices de runs pour éviter la duplication
entre `main.py` (CLI) et `notebook_principal.ipynb`.
"""

from __future__ import annotations

from typing import Any

from model_zoo import cuml_classic_deps_available, distilbert_deps_available

TEST_SIZE = 0.2
VAL_SIZE = 0.1
CV_FOLDS_STANDARD = 5
SCORING = "f1_macro"
RANDOM_STATE = 42
MAX_SAMPLES_DEFAULT = None

SELECTION_WEIGHTS_BALANCED = (0.30, 0.35, 0.20, 0.15)
SELECTION_WEIGHTS_CV_HEAVY = (0.25, 0.25, 0.35, 0.15)
SELECTION_WEIGHTS_DISTILBERT_SAFE = (0.30, 0.40, 0.15, 0.15)
SELECTION_WEIGHTS_TEST_PRIORITY = (0.25, 0.45, 0.15, 0.15)

HATE_RECALL_FLOOR = 0.40
HATE_RECALL_PENALTY = 0.03
DISTILBERT_PROXY_PENALTY_DEFAULT = 0.02

BASE_MODEL_NAMES = [
    "NaiveBayes",
    "LogisticRegression",
    "LinearSVC",
    "KNN",
    "DecisionTree",
    "RandomForest",
    "AdaBoost",
    "MLPClassifier",
    "LogisticRegressionGPU",
    "LinearSVCGPU",
    "KNNGPU",
    "RandomForestGPU",
    "DistilBERT",
]

GPU_CLASSIC_MODELS = {
    "LogisticRegressionGPU",
    "LinearSVCGPU",
    "KNNGPU",
    "RandomForestGPU",
}

ALGORITHM_SWITCHES_ALL = dict.fromkeys(BASE_MODEL_NAMES, True)
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
DISTILBERT_PARAMS_EP2 = {
    "DistilBERT": {
        "epochs": 2,
        "batch_size": 16,
        "max_length": 160,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
    }
}
DISTILBERT_RUN_MAP = {
    "run_d_fast": "fast",
    "run_d_balanced": "balanced",
    "run_d_robust": "robust",
    "run_d_vram_max": "vram_max",
}

MLP_RUN_MAP = {
    "run_mlp_fast": "fast",
    "run_mlp_balanced": "balanced",
    "run_mlp_wide": "wide",
    "run_mlp_ultra": "ultra",
}

ADABOOST_RUN_MAP = {
    "run_ada_fast": "fast",
    "run_ada_balanced": "balanced",
    "run_ada_wide": "wide",
    "run_ada_ultra": "ultra",
}


def _single_model_switches(target_model: str) -> dict[str, bool]:
    """Active uniquement le modèle ciblé dans un run."""
    return {name: (name == target_model) for name in ALGORITHM_SWITCHES_ALL}


def get_default_runs() -> dict[str, dict[str, Any]]:
    """Construit une matrice compacte de runs (coût maîtrisé)."""
    return {
        "run_a_data_balance": {
            "why": "Baseline robuste sur 100% des données, modèles classiques + GPU + DistilBERT.",
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
        "run_e_method_strict_classic": {
            "why": "Sélection méthodologique stricte en mode classique (sans DistilBERT).",
            "config": {
                "max_samples": MAX_SAMPLES_DEFAULT,
                "distilbert_epochs": 1,
                "include_distilbert": False,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "cv_folds": CV_FOLDS_STANDARD,
                "scoring": SCORING,
                "model_param_overrides": MODEL_PARAM_OVERRIDES_BASE,
                "model_grid_overrides": MODEL_GRID_OVERRIDES_BASE,
                "selection_weights": SELECTION_WEIGHTS_BALANCED,
                "algorithm_switches": ALGORITHM_SWITCHES_CLASSIC_ONLY,
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        },
        "run_g_distilbert_safe_ep3": {
            "why": "Run DistilBERT ep3 avec pondération adaptée au CV proxy.",
            "config": {
                "max_samples": MAX_SAMPLES_DEFAULT,
                "distilbert_epochs": 2,
                "include_distilbert": True,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "cv_folds": CV_FOLDS_STANDARD,
                "scoring": SCORING,
                "model_param_overrides": DISTILBERT_PARAMS_EP2,
                "model_grid_overrides": MODEL_GRID_OVERRIDES_BASE,
                "selection_weights": SELECTION_WEIGHTS_BALANCED,
                "algorithm_switches": ALGORITHM_SWITCHES_ALL,
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        },
    }


def _validate_profile_names(profile_names: tuple[str, ...] | None, available_profiles: dict[str, Any], family: str) -> tuple[str, ...]:
    """Valide une liste optionnelle de profils et retourne la sélection effective."""
    if profile_names is None:
        return tuple(available_profiles.keys())
    unknown = [name for name in profile_names if name not in available_profiles]
    if unknown:
        raise ValueError(f"Profils inconnus pour {family}: {unknown}. Disponibles: {list(available_profiles)}")
    return tuple(profile_names)


def get_available_exhaustive_options() -> dict[str, list[str]]:
    """Expose les options configurables de la matrice exhaustive (notebook/CLI)."""
    return {
        "distilbert_profiles": ["fast", "balanced", "robust", "vram_max"],
        "mlp_profiles": ["fast", "balanced", "wide", "ultra"],
        "adaboost_profiles": ["fast", "balanced", "wide", "ultra"],
        "gpu_profiles": ["gpu_fast", "gpu_balanced", "gpu_aggressive"],
        "gpu_models": sorted(GPU_CLASSIC_MODELS),
    }


def build_active_runs(
    run_matrix: str = "exhaustive",
    include_baseline: bool = True,
    distilbert_profile_names: tuple[str, ...] | None = None,
    mlp_profile_names: tuple[str, ...] | None = None,
    adaboost_profile_names: tuple[str, ...] | None = None,
    gpu_profile_names: tuple[str, ...] | None = None,
    gpu_model_names: tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Construit la matrice active puis filtre les runs incompatibles.

    Cette fonction est pensée pour réduire la logique dans le notebook/CLI:
    un seul point d'appel pour obtenir `(runs_actifs, runs_ignores)`.
    """
    if run_matrix == "default":
        runs = get_default_runs()
    elif run_matrix == "exhaustive":
        runs = get_exhaustive_runs(
            include_baseline=include_baseline,
            distilbert_profile_names=distilbert_profile_names,
            mlp_profile_names=mlp_profile_names,
            adaboost_profile_names=adaboost_profile_names,
            gpu_profile_names=gpu_profile_names,
            gpu_model_names=gpu_model_names,
        )
    else:
        raise ValueError("run_matrix doit être 'default' ou 'exhaustive'.")
    return filter_incompatible_runs(runs)


def get_exhaustive_runs(
    include_baseline: bool = True,
    distilbert_profile_names: tuple[str, ...] | None = None,
    mlp_profile_names: tuple[str, ...] | None = None,
    adaboost_profile_names: tuple[str, ...] | None = None,
    gpu_profile_names: tuple[str, ...] | None = None,
    gpu_model_names: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Construit une matrice explicite (1 run = 1 profil modèle).

    Les paramètres permettent d'activer seulement certains profils dans le notebook
    sans modifier le code des profils eux-mêmes.
    """
    exhaustive_runs: dict[str, dict[str, Any]] = {
        "run_a_data_balance": get_default_runs()["run_a_data_balance"]
    } if include_baseline else {}

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
    gpu_profiles: dict[str, dict[str, dict[str, list[Any]]]] = {
        "gpu_fast": {
            "LogisticRegressionGPU": {"svd__n_components": [64], "clf__C": [1.0]},
            "LinearSVCGPU": {"svd__n_components": [64], "clf__C": [1.0]},
            "KNNGPU": {"svd__n_components": [64], "clf__n_neighbors": [11], "clf__weights": ["uniform"]},
            "RandomForestGPU": {"svd__n_components": [96], "clf__n_estimators": [300], "clf__max_depth": [16], "clf__max_features": [0.8]},
        },
        "gpu_balanced": {
            "LogisticRegressionGPU": {"svd__n_components": [64, 128], "clf__C": [0.5, 1.0, 2.0]},
            "LinearSVCGPU": {"svd__n_components": [64, 128], "clf__C": [0.5, 1.0, 2.0]},
            "KNNGPU": {"svd__n_components": [64, 128], "clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform"]},
            "RandomForestGPU": {"svd__n_components": [96, 160], "clf__n_estimators": [300, 600], "clf__max_depth": [16, 24], "clf__max_features": [0.5, 0.8, 1.0]},
        },
        "gpu_aggressive": {
            "LogisticRegressionGPU": {"tfidf__ngram_range": [(1, 1), (1, 2)], "svd__n_components": [64, 128, 192], "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0]},
            "LinearSVCGPU": {"tfidf__ngram_range": [(1, 1), (1, 2)], "svd__n_components": [64, 128, 192], "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0]},
            "KNNGPU": {"svd__n_components": [64, 128, 192], "clf__n_neighbors": [3, 5, 11, 21, 31], "clf__weights": ["uniform"]},
            "RandomForestGPU": {"svd__n_components": [96, 160, 224], "clf__n_estimators": [300, 600, 900], "clf__max_depth": [16, 24, 32], "clf__max_features": [0.4, 0.6, 0.8, 1.0]},
        },
    }

    selected_distilbert_profiles = _validate_profile_names(distilbert_profile_names, distilbert_profiles, "DistilBERT")
    selected_mlp_profiles = _validate_profile_names(mlp_profile_names, mlp_profiles, "MLPClassifier")
    selected_adaboost_profiles = _validate_profile_names(adaboost_profile_names, adaboost_profiles, "AdaBoost")
    selected_gpu_profiles = _validate_profile_names(gpu_profile_names, gpu_profiles, "GPU")
    selected_gpu_models = tuple(gpu_model_names) if gpu_model_names is not None else tuple(sorted(GPU_CLASSIC_MODELS))
    invalid_gpu_models = [name for name in selected_gpu_models if name not in GPU_CLASSIC_MODELS]
    if invalid_gpu_models:
        raise ValueError(f"Modèles GPU inconnus: {invalid_gpu_models}. Disponibles: {sorted(GPU_CLASSIC_MODELS)}")

    for run_name, profile_name in DISTILBERT_RUN_MAP.items():
        if profile_name not in selected_distilbert_profiles:
            continue
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

    for run_name, profile_name in MLP_RUN_MAP.items():
        if profile_name not in selected_mlp_profiles:
            continue
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
                "model_grid_overrides": mlp_profiles[profile_name],
                "selection_weights": SELECTION_WEIGHTS_CV_HEAVY,
                "algorithm_switches": _single_model_switches("MLPClassifier"),
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        }

    for run_name, profile_name in ADABOOST_RUN_MAP.items():
        if profile_name not in selected_adaboost_profiles:
            continue
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
                "model_grid_overrides": adaboost_profiles[profile_name],
                "selection_weights": SELECTION_WEIGHTS_TEST_PRIORITY,
                "algorithm_switches": _single_model_switches("AdaBoost"),
                "hate_recall_floor": HATE_RECALL_FLOOR,
                "hate_recall_penalty": HATE_RECALL_PENALTY,
                "random_state": RANDOM_STATE,
            },
        }

    for gpu_model_name in selected_gpu_models:
        for profile_name in selected_gpu_profiles:
            run_name = f"run_{gpu_model_name.lower()}_{profile_name}"
            exhaustive_runs[run_name] = {
                "why": f"Run tuning {gpu_model_name} seul (profil={profile_name}).",
                "config": {
                    "max_samples": MAX_SAMPLES_DEFAULT,
                    "distilbert_epochs": 1,
                    "include_distilbert": False,
                    "test_size": TEST_SIZE,
                    "val_size": VAL_SIZE,
                    "cv_folds": CV_FOLDS_STANDARD,
                    "scoring": SCORING,
                    "model_param_overrides": MODEL_PARAM_OVERRIDES_BASE,
                    "model_grid_overrides": {gpu_model_name: gpu_profiles[profile_name][gpu_model_name]},
                    "selection_weights": SELECTION_WEIGHTS_CV_HEAVY,
                    "algorithm_switches": _single_model_switches(gpu_model_name),
                    "hate_recall_floor": HATE_RECALL_FLOOR,
                    "hate_recall_penalty": HATE_RECALL_PENALTY,
                    "random_state": RANDOM_STATE,
                },
            }

    return exhaustive_runs


def _enabled_models_from_switches(switches: dict[str, Any]) -> set[str]:
    """Extrait l'ensemble des modèles activés à partir de `algorithm_switches`."""
    return {name for name, enabled in switches.items() if bool(enabled)}


def filter_incompatible_runs(runs: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Retire les runs incompatibles avec l'environnement courant.

    - DistilBERT-only est retiré si les dépendances deep ne sont pas présentes.
    - GPU-only (cuML) est retiré si les dépendances GPU classiques ne sont pas présentes.
    """
    has_distilbert = distilbert_deps_available()
    has_cuml = cuml_classic_deps_available()

    filtered: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for run_name, run_ctx in runs.items():
        config = run_ctx.get("config", {})
        switches = config.get("algorithm_switches", {})
        if not isinstance(switches, dict):
            filtered[run_name] = run_ctx
            continue

        enabled_models = _enabled_models_from_switches(switches)
        if enabled_models == {"DistilBERT"} and not has_distilbert:
            skipped.append(run_name)
            continue
        if enabled_models and enabled_models.issubset(GPU_CLASSIC_MODELS) and not has_cuml:
            skipped.append(run_name)
            continue
        filtered[run_name] = run_ctx
    return filtered, skipped
