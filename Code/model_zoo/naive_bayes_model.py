"""Définition du modèle Naive Bayes."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .spec import ModelSpec


class NaiveBayesModel:
    """Fabrique du modèle Naive Bayes pour texte TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:  # noqa: ARG004
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", MultinomialNB()),
            ],
            memory=None,
        )
        return ModelSpec(
            name="NaiveBayes",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__alpha": [0.5, 1.0],
            },
            why="Baseline rapide en texte; robuste avec des comptes/poids TF-IDF.",
        )
