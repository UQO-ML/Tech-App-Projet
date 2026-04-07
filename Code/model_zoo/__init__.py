"""Exports du zoo de modèles."""

from .adaboost_model import AdaBoostModel
from .cuml_wrappers import cuml_classic_deps_available
from .decision_tree_model import DecisionTreeModel
from .distilbert_model import DistilBertTextClassifier, build_distilbert_tuning, distilbert_deps_available
from .knn_model import KNNModel
from .knn_gpu_model import KNNGPUModel
from .linear_svc_model import LinearSVCModel
from .linear_svc_gpu_model import LinearSVCGPUModel
from .logistic_regression_model import LogisticRegressionModel
from .logistic_regression_gpu_model import LogisticRegressionGPUModel
from .mlp_model import MLPModel
from .naive_bayes_model import NaiveBayesModel
from .random_forest_model import RandomForestModel
from .random_forest_gpu_model import RandomForestGPUModel
from .spec import ModelSpec

CLASSIC_MODEL_BUILDERS = [
    NaiveBayesModel,
    LogisticRegressionModel,
    LinearSVCModel,
    KNNModel,
    DecisionTreeModel,
    RandomForestModel,
    AdaBoostModel,
    MLPModel,
    LogisticRegressionGPUModel,
    LinearSVCGPUModel,
    KNNGPUModel,
    RandomForestGPUModel,
]

__all__ = [
    "CLASSIC_MODEL_BUILDERS",
    "DistilBertTextClassifier",
    "ModelSpec",
    "build_distilbert_tuning",
    "cuml_classic_deps_available",
    "distilbert_deps_available",
]
