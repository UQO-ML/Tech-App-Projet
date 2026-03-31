"""Définition et entraînement de plusieurs modèles de classification texte."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from utils import compute_metrics


def get_models(random_state: int = 42) -> dict[str, dict[str, Any]]:
    """Retourne un ensemble de modèles pertinents pour du texte TF-IDF multi-classes."""
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
            "why": "Baseline rapide en texte; robuste avec des comptes/poids TF-IDF.",
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
            "why": "Très adaptée aux données textuelles clairsemées; souvent un excellent compromis performance/interprétabilité.",
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
            "why": "SVM linéaire performant sur grands espaces de features TF-IDF et classes parfois déséquilibrées.",
        },
        "KNN": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=12000)),
                    ("clf", KNeighborsClassifier(n_neighbors=5)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__n_neighbors": [3, 5, 11],
                "clf__weights": ["uniform", "distance"],
            },
            "why": "Référence non paramétrique utile comme comparaison, même si souvent moins performante sur texte sparse.",
        },
        "DecisionTree": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=6000)),
                    ("clf", DecisionTreeClassifier(random_state=random_state, class_weight="balanced", ccp_alpha=0.0)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__max_depth": [20, 40, None],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__ccp_alpha": [0.0, 0.001],
            },
            "why": "Modèle interprétable pour analyser des règles décisionnelles, pertinent comme repère explicatif.",
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
            "why": "Ensemble d'arbres plus stable qu'un arbre simple; capture des interactions non linéaires.",
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


def get_model_rationales(random_state: int = 42) -> dict[str, str]:
    """Retourne le rationnel 'pourquoi ce modèle' pour le rapport."""
    return {name: entry.get("why", "") for name, entry in get_models(random_state=random_state).items()}
