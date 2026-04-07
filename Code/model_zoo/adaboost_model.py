"""Définition du modèle AdaBoost."""

from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .spec import ModelSpec


class AdaBoostModel:
    """Fabrique du modèle AdaBoost pour texte TF-IDF."""

    @staticmethod
    def build_spec(random_state: int) -> ModelSpec:
        """Construit la spec modèle + grille.

        Paramètres:
            random_state: Seed de reproductibilité.

        Retour:
            Spécification complète d'AdaBoost (pipeline, grille, justification).
        """
        weak_tree = DecisionTreeClassifier(
            max_depth=1,
            min_samples_leaf=1,
            class_weight="balanced",
            ccp_alpha=0.0,
            random_state=random_state,
        )
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=12000)),
                (
                    "clf",
                    AdaBoostClassifier(
                        estimator=weak_tree,
                        n_estimators=300,
                        learning_rate=0.1,
                        random_state=random_state,
                    ),
                ),
            ],
            memory=None,
        )
        return ModelSpec(
            name="AdaBoost",
            pipeline=pipeline,
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__n_estimators": [200, 400, 800],
                "clf__learning_rate": [0.03, 0.1, 0.3, 1.0],
                "clf__estimator__max_depth": [1, 2, 3],
                "clf__estimator__min_samples_leaf": [1, 2],
            },
            why=(
                "Boosting séquentiel de classifieurs faibles pour réduire le biais; "
                "peut capturer des frontières plus complexes qu'un arbre unique."
            ),
        )
