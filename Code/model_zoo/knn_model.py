"""Définition du modèle KNN."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .spec import ModelSpec


class KNNModel:
    """Fabrique du modèle KNN pour texte TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:  # noqa: ARG004
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=12000)),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ],
            memory=None,
        )
        return ModelSpec(
            name="KNN",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__n_neighbors": [3, 5, 11],
                "clf__weights": ["uniform", "distance"],
            },
            why="Référence non paramétrique utile comme comparaison.",
        )
