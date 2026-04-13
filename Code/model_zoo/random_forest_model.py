"""Définition du modèle Random Forest."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .spec import ModelSpec


class RandomForestModel:
    """Fabrique du modèle Random Forest pour texte TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
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
        )
        return ModelSpec(
            name="RandomForest",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 30],
                "clf__max_features": ["sqrt", "log2"],
                "clf__min_samples_leaf": [1, 2],
            },
            why="Ensemble d'arbres plus stable qu'un arbre simple; capture des interactions non linéaires.",
        )
