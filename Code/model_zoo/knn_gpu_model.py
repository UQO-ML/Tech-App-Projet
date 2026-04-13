"""Définition du modèle KNN GPU (cuML)."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cuml_wrappers import CumlKNNClassifier
from .spec import ModelSpec


class KNNGPUModel:
    """Fabrique un KNN GPU sur projection dense SVD."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=18000)),
                ("svd", TruncatedSVD(random_state=random_state)),
                ("scaler", StandardScaler()),
                ("clf", CumlKNNClassifier()),
            ],
            memory=None,
        )
        return ModelSpec(
            name="KNNGPU",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "svd__n_components": [64, 128],
                "clf__n_neighbors": [5, 11, 21],
                # `distance` peut échouer selon le toolchain CUDA/NVRTC.
                "clf__weights": ["uniform"],
            },
            why=(
                "KNN GPU via cuML: souvent plus rapide que la version CPU sur des embeddings "
                "denses, utile pour explorer des profils voisins à grand volume."
            ),
        )
