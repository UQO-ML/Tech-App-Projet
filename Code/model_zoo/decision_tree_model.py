"""Définition du modèle Decision Tree."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .spec import ModelSpec


class DecisionTreeModel:
    """Fabrique du modèle Decision Tree pour texte TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=6000)),
                ("clf", DecisionTreeClassifier(random_state=random_state, class_weight="balanced", ccp_alpha=0.0)),
            ],
            memory=None,
        )
        return ModelSpec(
            name="DecisionTree",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__max_depth": [20, 40, None],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__ccp_alpha": [0.0, 0.001],
            },
            why="Modèle interprétable pour analyser des règles décisionnelles.",
        )
