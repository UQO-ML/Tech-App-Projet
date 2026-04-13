"""Définition du modèle Linear SVC GPU (cuML)."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cuml_wrappers import CumlLinearSVCClassifier
from .spec import ModelSpec


class LinearSVCGPUModel:
    """Fabrique un SVM linéaire GPU sur projection dense SVD."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20000)),
                ("svd", TruncatedSVD(random_state=random_state)),
                ("scaler", StandardScaler()),
                ("clf", CumlLinearSVCClassifier(random_state=random_state)),
            ],
            memory=None,
        )
        return ModelSpec(
            name="LinearSVCGPU",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "svd__n_components": [64, 128],
                "clf__C": [0.5, 1.0, 2.0],
            },
            why=(
                "Version GPU du classifieur linéaire (cuML), pertinente pour des comparaisons "
                "directes avec LinearSVC sklearn sur de gros lots de runs."
            ),
        )
