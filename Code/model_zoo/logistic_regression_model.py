"""Définition du modèle Logistic Regression."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .spec import ModelSpec


class LogisticRegressionModel:
    """Fabrique du modèle Logistic Regression pour texte TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ],
            memory=None,
        )
        return ModelSpec(
            name="LogisticRegression",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__C": [0.5, 1.0, 2.0],
            },
            why="Très adaptée aux données textuelles clairsemées; excellent compromis performance/interprétabilité.",
        )
