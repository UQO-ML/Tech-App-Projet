"""Définition du modèle MLPClassifier."""

from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .spec import ModelSpec


class MLPModel:
    """Fabrique un MLP dense sur représentation SVD de TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille."""
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=15000)),
                ("svd", TruncatedSVD(random_state=random_state)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        random_state=random_state,
                        max_iter=220,
                        early_stopping=True,
                    ),
                ),
            ],
            memory=None,
        )
        return ModelSpec(
            name="MLPClassifier",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                # Composantes modérées pour rester valides sur petits folds.
                "svd__n_components": [20, 40],
                "clf__hidden_layer_sizes": [(128,), (256, 128)],
                "clf__alpha": [1e-4, 1e-3],
                "clf__learning_rate_init": [1e-3, 5e-4],
            },
            why="Réseau de neurones dense classique pour satisfaire explicitement l'exigence NN.",
        )
