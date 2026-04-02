"""Orchestration de l'entraînement des modèles de classification."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, cross_val_score

from model_zoo import CLASSIC_MODEL_BUILDERS
from model_zoo import DistilBertTextClassifier, build_distilbert_tuning, distilbert_deps_available
from utils import compute_metrics


def resolve_algorithm_switches(
    include_distilbert: bool = True,
    algorithm_switches: dict[str, bool] | None = None,
    random_state: int = 42,
) -> dict[str, bool]:
    """Résout l'état activé/désactivé de chaque algorithme.

    Paramètres:
        include_distilbert: Compatibilité historique pour DistilBERT.
        algorithm_switches: Dictionnaire optionnel `{nom_modele: bool}`.
        random_state: Seed (utilisée pour récupérer les noms des modèles classiques).

    Retour:
        Dictionnaire complet `{nom_modele: actif}` incluant DistilBERT.
    """
    switches = dict.fromkeys(get_models(random_state=random_state), True)
    switches["DistilBERT"] = bool(include_distilbert)

    if algorithm_switches:
        for model_name, is_enabled in algorithm_switches.items():
            if model_name in switches:
                switches[model_name] = bool(is_enabled)
    return switches


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


def _merge_param_grid(base_grid: dict[str, list[Any]], override_grid: dict[str, list[Any]] | None) -> dict[str, list[Any]]:
    """Fusionne une grille par défaut avec une surcharge partielle.

    Paramètres:
        base_grid: Grille d'origine définie dans le model zoo.
        override_grid: Dictionnaire de surcharge pour un modèle donné.
            - clé: nom de paramètre sklearn (`clf__alpha`, `svd__n_components`, etc.);
            - valeur: liste de valeurs à tester.
            Les clés absentes conservent la grille par défaut.

    Retour:
        Grille effective utilisée par GridSearchCV.
    """
    merged = dict(base_grid)
    if not override_grid:
        return merged
    for param_name, candidate_values in override_grid.items():
        if isinstance(candidate_values, list) and candidate_values:
            merged[param_name] = candidate_values
    return merged


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


def get_expected_model_names(
    include_distilbert: bool = True,
    algorithm_switches: dict[str, bool] | None = None,
    random_state: int = 42,
) -> list[str]:
    """Retourne la liste des modèles activés à inclure dans le report final.

    Paramètres:
        include_distilbert: Compatibilité historique pour DistilBERT.
        algorithm_switches: Dictionnaire optionnel `{nom_modele: bool}`.
        random_state: Seed (utilisée pour récupérer les noms des modèles classiques).

    Retour:
        Liste ordonnée des modèles activés.
    """
    switches = resolve_algorithm_switches(
        include_distilbert=include_distilbert,
        algorithm_switches=algorithm_switches,
        random_state=random_state,
    )
    ordered_names = list(get_models(random_state=random_state).keys()) + ["DistilBERT"]
    return [name for name in ordered_names if switches.get(name, False)]


def train_with_grid_search(
    x_train,
    y_train,
    model_name: str,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "f1_macro",
    grid_override: dict[str, list[Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Optimise un modèle sklearn via GridSearchCV.

    Paramètres:
        x_train: Textes d'entraînement.
        y_train: Labels d'entraînement.
        model_name: Nom du modèle classique à entraîner.
        random_state: Seed de reproductibilité.
        cv: Nombre de folds CV.
            - 3: plus rapide, moins stable;
            - 5: compromis recommandé;
            - >5: plus coûteux mais plus robuste.
        scoring: Métrique sklearn pour sélectionner les hyperparamètres.
            Exemple recommandé: `f1_macro`.
        grid_override: Surcharge partielle de grille pour le modèle ciblé.
            Permet d'explorer des plages plus agressives sans modifier le model zoo.

    Retour:
        Tuple `(best_estimator, tuning_info)`.
    """
    model_entry = get_models(random_state=random_state)[model_name]
    effective_grid = _merge_param_grid(model_entry["param_grid"], grid_override)
    search = GridSearchCV(
        estimator=model_entry["pipeline"],
        param_grid=effective_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "effective_param_grid": effective_grid,
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
    model_grid_overrides: dict[str, dict[str, list[Any]]] | None = None,
) -> dict[str, Any]:
    """Entraîne un modèle classique et retourne un résultat normalisé.

    Paramètres:
        model_name: Nom du modèle classique.
        model_grid_overrides: Grilles optionnelles par modèle.
            Exemple:
            `{"MLPClassifier": {"clf__alpha": [1e-4, 1e-3]}}`.
    """
    grid_override = (model_grid_overrides or {}).get(model_name)
    estimator, tuning_info = train_with_grid_search(
        x_train=x_train,
        y_train=y_train,
        model_name=model_name,
        random_state=random_state,
        cv=cv_folds,
        scoring=scoring,
        grid_override=grid_override,
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
    distilbert_param_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Entraîne DistilBERT, sinon retourne un statut `skipped`/`failed` explicite.

    Paramètres:
        distilbert_epochs: Valeur historique d'epochs (compatibilité).
        distilbert_param_overrides: Paramètres optionnels passés au classifieur.
            Clés utiles:
            - `epochs` (1-5 recommandé): plus élevé = meilleure adaptation, plus coûteux.
            - `batch_size` (8/16/32): plus petit = moins de mémoire, entraînement plus lent.
            - `max_length` (96-256): plus grand = plus de contexte, plus de coût mémoire/temps.
            - `learning_rate` (2e-5 à 5e-5): trop haut instable, trop bas sous-apprentissage.
            - `weight_decay` (0.0 à 0.1): régularisation.
    """
    if not distilbert_deps_available():
        return {
            "status": "skipped",
            "estimator": None,
            "val_metrics": {},
            "tuning": {},
            "error": "Dépendances manquantes: torch/transformers/datasets.",
        }
    estimator_kwargs = {
        "random_state": random_state,
        "epochs": distilbert_epochs,
        "batch_size": 16,
        "max_length": 128,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
    }
    if distilbert_param_overrides:
        estimator_kwargs.update(distilbert_param_overrides)
    estimator = DistilBertTextClassifier(
        **estimator_kwargs,
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
    algorithm_switches: dict[str, bool] | None = None,
    distilbert_epochs: int = 1,
    cv_folds: int = 5,
    scoring: str = "f1_macro",
    model_param_overrides: dict[str, dict[str, Any]] | None = None,
    model_grid_overrides: dict[str, dict[str, list[Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Entraîne tous les modèles demandés (classiques + DistilBERT optionnel).

    Paramètres:
        model_param_overrides: Paramètres fixes par modèle.
            Usage principal: DistilBERT (`epochs`, `batch_size`, `max_length`, ...).
        model_grid_overrides: Surcouches de grilles GridSearch par modèle classique.
            Impact:
            - plus de combinaisons -> meilleure chance d'optimum, mais coût CPU/RAM plus élevé;
            - plages restreintes -> exécution plus rapide, exploration moins large.
    """
    switches = resolve_algorithm_switches(
        include_distilbert=include_distilbert,
        algorithm_switches=algorithm_switches,
        random_state=random_state,
    )
    results: dict[str, dict[str, Any]] = {}
    for model_name in get_models(random_state=random_state):
        if not switches.get(model_name, True):
            continue
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
                model_grid_overrides=model_grid_overrides,
            )
        except Exception as exc:
            results[model_name] = _build_failed_result(str(exc))

    if switches.get("DistilBERT", False):
        try:
            results["DistilBERT"] = _train_or_skip_distilbert(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                random_state=random_state,
                distilbert_epochs=distilbert_epochs,
                distilbert_param_overrides=(model_param_overrides or {}).get("DistilBERT"),
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


def get_model_rationales(
    random_state: int = 42,
    include_distilbert: bool = True,
    algorithm_switches: dict[str, bool] | None = None,
) -> dict[str, str]:
    """Retourne un texte court de justification pour les modèles activés.

    Paramètres:
        random_state: Seed utilisée pour récupérer les specs.
        include_distilbert: Compatibilité historique pour DistilBERT.
        algorithm_switches: Dictionnaire optionnel `{nom_modele: bool}`.

    Retour:
        Dictionnaire `{nom_modele: justification}` pour les modèles actifs.
    """
    switches = resolve_algorithm_switches(
        include_distilbert=include_distilbert,
        algorithm_switches=algorithm_switches,
        random_state=random_state,
    )
    rationales = {
        name: entry.get("why", "")
        for name, entry in get_models(random_state=random_state).items()
        if switches.get(name, False)
    }
    if switches.get("DistilBERT", False):
        rationales["DistilBERT"] = (
            "Modèle Transformer pré-entraîné qui capte mieux le contexte sémantique des tweets; "
            "souvent plus performant sur la détection de nuances offensantes/haineuses."
        )
    return rationales
