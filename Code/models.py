"""Orchestration de l'entraînement des modèles de classification."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, cross_val_score

from model_zoo import CLASSIC_MODEL_BUILDERS
from model_zoo import DistilBertTextClassifier, build_distilbert_tuning, distilbert_deps_available
from utils import compute_metrics


def get_models(random_state: int = 42) -> dict[str, dict[str, Any]]:
    """Retourne toutes les specs de modèles classiques."""
    models: dict[str, dict[str, Any]] = {}
    for builder in CLASSIC_MODEL_BUILDERS:
        spec = builder.build_spec(random_state=random_state)
        models[spec.name] = {
            "pipeline": spec.pipeline,
            "param_grid": spec.param_grid,
            "why": spec.why,
        }
    return models


def _build_failed_result(error_message: str) -> dict[str, Any]:
    """Construit une structure de résultat standard en cas d'échec."""
    return {
        "status": "failed",
        "estimator": None,
        "val_metrics": {},
        "tuning": {},
        "error": error_message,
    }


def _build_trained_result(estimator: Any, y_val, y_val_pred, tuning: dict[str, Any]) -> dict[str, Any]:
    """Construit une structure de résultat standard après entraînement réussi."""
    return {
        "status": "trained",
        "estimator": estimator,
        "val_metrics": compute_metrics(y_val, y_val_pred),
        "tuning": tuning,
        "error": None,
    }


def get_expected_model_names(include_distilbert: bool = True) -> list[str]:
    """Retourne la liste des modèles qui doivent apparaître dans le report final."""
    names = list(get_models().keys())
    if include_distilbert:
        names.append("DistilBERT")
    return names


def train_with_grid_search(
    x_train,
    y_train,
    model_name: str,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "f1_macro",
) -> tuple[Any, dict[str, Any]]:
    """Optimise un modèle sklearn via GridSearchCV."""
    model_entry = get_models(random_state=random_state)[model_name]
    search = GridSearchCV(
        estimator=model_entry["pipeline"],
        param_grid=model_entry["param_grid"],
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
    }


def _train_single_classic_model(
    model_name: str,
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int,
    cv_folds: int,
    scoring: str,
) -> dict[str, Any]:
    """Entraîne un modèle classique et retourne un résultat normalisé."""
    estimator, tuning_info = train_with_grid_search(
        x_train=x_train,
        y_train=y_train,
        model_name=model_name,
        random_state=random_state,
        cv=cv_folds,
        scoring=scoring,
    )
    y_val_pred = estimator.predict(x_val)
    return _build_trained_result(estimator=estimator, y_val=y_val, y_val_pred=y_val_pred, tuning=tuning_info)


def _train_or_skip_distilbert(
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int,
    distilbert_epochs: int,
) -> dict[str, Any]:
    """Entraîne DistilBERT, sinon retourne un statut `skipped`/`failed` explicite."""
    if not distilbert_deps_available():
        return {
            "status": "skipped",
            "estimator": None,
            "val_metrics": {},
            "tuning": {},
            "error": "Dépendances manquantes: torch/transformers/datasets.",
        }
    estimator = DistilBertTextClassifier(
        random_state=random_state,
        epochs=distilbert_epochs,
        batch_size=16,
        max_length=128,
    )
    estimator.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    y_val_pred = estimator.predict(x_val)
    return _build_trained_result(
        estimator=estimator,
        y_val=y_val,
        y_val_pred=y_val_pred,
        tuning=build_distilbert_tuning(estimator),
    )


def train_all_models(
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int = 42,
    include_distilbert: bool = True,
    distilbert_epochs: int = 1,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
) -> dict[str, dict[str, Any]]:
    """Entraîne tous les modèles demandés (classiques + DistilBERT optionnel)."""
    results: dict[str, dict[str, Any]] = {}
    for model_name in get_models(random_state=random_state):
        try:
            results[model_name] = _train_single_classic_model(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                random_state=random_state,
                cv_folds=cv_folds,
                scoring=scoring,
            )
        except Exception as exc:
            results[model_name] = _build_failed_result(str(exc))

    if include_distilbert:
        try:
            results["DistilBERT"] = _train_or_skip_distilbert(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                random_state=random_state,
                distilbert_epochs=distilbert_epochs,
            )
        except Exception as exc:
            results["DistilBERT"] = _build_failed_result(str(exc))

    return results


def cross_validate_estimator(estimator, x, y, cv: int = 5) -> list[float]:
    """Retourne les scores F1 macro en validation croisée."""
    if getattr(estimator, "skip_cv", False):
        return []
    scores = cross_val_score(estimator, x, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return [float(score) for score in scores]


def get_model_rationales(random_state: int = 42) -> dict[str, str]:
    """Retourne un texte court de justification pour chaque modèle."""
    rationales = {name: entry.get("why", "") for name, entry in get_models(random_state=random_state).items()}
    rationales["DistilBERT"] = (
        "Modèle Transformer pré-entraîné qui capte mieux le contexte sémantique des tweets; "
        "souvent plus performant sur la détection de nuances offensantes/haineuses."
    )
    return rationales
