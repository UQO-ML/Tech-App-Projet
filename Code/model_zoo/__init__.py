"""Exports du zoo de modèles."""

from .decision_tree_model import DecisionTreeModel
from .distilbert_model import DistilBertTextClassifier, build_distilbert_tuning, distilbert_deps_available
from .knn_model import KNNModel
from .linear_svc_model import LinearSVCModel
from .logistic_regression_model import LogisticRegressionModel
from .mlp_model import MLPModel
from .naive_bayes_model import NaiveBayesModel
from .random_forest_model import RandomForestModel
from .spec import ModelSpec

CLASSIC_MODEL_BUILDERS = [
    NaiveBayesModel,
    LogisticRegressionModel,
    LinearSVCModel,
    KNNModel,
    DecisionTreeModel,
    RandomForestModel,
    MLPModel,
]

__all__ = [
    "CLASSIC_MODEL_BUILDERS",
    "DistilBertTextClassifier",
    "ModelSpec",
    "build_distilbert_tuning",
    "distilbert_deps_available",
]
