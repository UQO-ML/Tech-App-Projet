"""Définition et entraînement de plusieurs modèles de classification texte."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import compute_metrics


def get_models(random_state: int = 42) -> dict[str, dict[str, Any]]:
    """Retourne 4 pipelines texte + grilles d'hyperparamètres."""
    return {
        "NaiveBayes": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("clf", MultinomialNB()),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__alpha": [0.5, 1.0],
            },
        },
        "LogisticRegression": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__C": [0.5, 1.0, 2.0],
            },
        },
        "LinearSVC": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("clf", LinearSVC(class_weight="balanced", random_state=random_state)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__C": [0.5, 1.0, 2.0],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline(
                [
                    # On limite les features pour garder un temps de calcul raisonnable.
                    ("tfidf", TfidfVectorizer(max_features=8000)),
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=-1,
                            min_samples_leaf=1,
                            max_features="sqrt",
                        ),
                    ),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 30],
                "clf__max_features": ["sqrt", "log2"],
                "clf__min_samples_leaf": [1, 2],
            },
        },
    }


def train_with_grid_search(
    x_train,
    y_train,
    model_name: str,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "f1_macro",
) -> tuple[Any, dict[str, Any]]:
    """Optimise un modèle avec GridSearchCV puis retourne le meilleur estimateur."""
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


def train_all_models(
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Entraîne tous les modèles, puis calcule leurs métriques de validation."""
    results: dict[str, dict[str, Any]] = {}
    for model_name in get_models(random_state=random_state):
        estimator, tuning_info = train_with_grid_search(
            x_train,
            y_train,
            model_name=model_name,
            random_state=random_state,
        )
        y_val_pred = estimator.predict(x_val)
        results[model_name] = {
            "estimator": estimator,
            "val_metrics": compute_metrics(y_val, y_val_pred),
            "tuning": tuning_info,
        }
    return results


def cross_validate_estimator(estimator, X, y, cv: int = 5) -> list[float]:
    """Retourne les scores F1 macro de validation croisée k-fold."""
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return [float(score) for score in scores]


def select_best_model(results: dict[str, dict[str, Any]], metric_name: str = "f1_macro") -> tuple[str, dict[str, Any]]:
    """Sélectionne le meilleur modèle selon une métrique de validation."""
    best_name = max(results, key=lambda name: results[name]["val_metrics"][metric_name])
    return best_name, results[best_name]
