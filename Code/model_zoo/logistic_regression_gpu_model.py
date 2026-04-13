"""Définition du modèle Logistic Regression GPU (cuML)."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cuml_wrappers import CumlLogisticRegressionClassifier
from .spec import ModelSpec


class LogisticRegressionGPUModel:
    """Fabrique une Logistic Regression GPU sur projection dense SVD."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=20000)),
                ("svd", TruncatedSVD(random_state=random_state)),
                ("scaler", StandardScaler()),
                ("clf", CumlLogisticRegressionClassifier(random_state=random_state)),
            ],
            memory=None,
        )
        return ModelSpec(
            name="LogisticRegressionGPU",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "svd__n_components": [64, 128],
                "clf__C": [0.5, 1.0, 2.0],
            },
            why=(
                "Alternative GPU de Logistic Regression via cuML: utile pour accélérer les runs "
                "sur des représentations textuelles denses après réduction SVD."
            ),
        )
