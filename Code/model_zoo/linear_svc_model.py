"""Définition du modèle LinearSVC."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .spec import ModelSpec


class LinearSVCModel:
    """Fabrique du modèle SVM linéaire pour texte TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", LinearSVC(class_weight="balanced", random_state=random_state)),
            ],
            memory=None,
        )
        return ModelSpec(
            name="LinearSVC",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__C": [0.5, 1.0, 2.0],
            },
            why="SVM linéaire performant sur grands espaces de features TF-IDF.",
        )
