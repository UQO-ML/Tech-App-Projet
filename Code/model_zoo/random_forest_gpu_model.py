"""Définition du modèle Random Forest GPU (cuML)."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .cuml_wrappers import CumlRandomForestClassifier
from .spec import ModelSpec


class RandomForestGPUModel:
    """Fabrique un Random Forest GPU sur projection dense SVD."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20000)),
                ("svd", TruncatedSVD(random_state=random_state)),
                ("clf", CumlRandomForestClassifier(random_state=random_state)),
            ],
            memory=None,
        )
        return ModelSpec(
            name="RandomForestGPU",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "svd__n_components": [96, 160],
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [16, 24, 32],
                "clf__max_features": [0.5, 0.8, 1.0],
            },
            why=(
                "Version GPU de Random Forest via cuML: utile pour réduire le temps des grilles "
                "arbres tout en conservant une logique de modèle non linéaire."
            ),
        )
