"""Wrappers sklearn-compatibles pour estimateurs cuML.

Ce module encapsule les modèles GPU cuML derrière une API similaire à scikit-learn
afin de pouvoir les réutiliser dans les pipelines et GridSearchCV existants.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

GPU_DEPS_ERROR = (
    "Dépendances GPU classiques manquantes: installer `cuml` (et son stack CUDA) "
    "pour activer les modèles GPU de type Logistic/LinearSVC/KNN/RandomForest."
)
NOT_FITTED_ERROR = "Le modèle doit être entraîné avant predict()."


def cuml_classic_deps_available() -> bool:
    """Retourne True si les dépendances minimales cuML sont disponibles."""
    try:
        import cuml  # noqa: F401  # pyright: ignore[reportMissingImports]
        import cupy  # noqa: F401  # pyright: ignore[reportMissingImports]
        return True
    except Exception:
        return False


def _require_cuml_deps() -> None:
    """Lève une erreur explicite si cuML/cupy n'est pas installée."""
    if not cuml_classic_deps_available():
        raise ImportError(GPU_DEPS_ERROR)


def _dense_numpy(x: Any) -> np.ndarray:
    """Convertit les entrées vers `np.ndarray` dense, compatible cuML."""
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _to_numpy(y_pred: Any) -> np.ndarray:
    """Convertit la sortie cuML/cupy/cudf en numpy 1D."""
    if hasattr(y_pred, "to_numpy"):
        return np.asarray(y_pred.to_numpy())
    try:
        import cupy as cp  # pyright: ignore[reportMissingImports]

        if isinstance(y_pred, cp.ndarray):
            return cp.asnumpy(y_pred)
    except Exception:
        pass
    return np.asarray(y_pred)


def _normalize_labels(y: Any) -> np.ndarray:
    """Normalise les labels vers un vecteur numpy 1D."""
    y_np = np.asarray(y)
    if y_np.ndim > 1:
        y_np = y_np.ravel()
    return y_np


def _construct_with_supported_kwargs(estimator_cls: Any, kwargs: dict[str, Any]) -> Any:
    """Construit un estimateur en filtrant les kwargs non supportés."""
    signature = inspect.signature(estimator_cls)
    allowed = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in allowed and value is not None}
    return estimator_cls(**filtered)


class CumlLogisticRegressionClassifier(ClassifierMixin, BaseEstimator):
    """Logistic Regression GPU via cuML, compatible pipeline sklearn."""

    skip_cv = True
    is_gpu_model = True

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None

    def fit(self, x, y):
        """Entraîne la Logistic Regression GPU sur des features denses."""
        _require_cuml_deps()
        from cuml.linear_model import LogisticRegression as CumlLogisticRegression  # pyright: ignore[reportMissingImports]
        x_np = _dense_numpy(x)
        y_np = _normalize_labels(y)

        self._model = _construct_with_supported_kwargs(
            CumlLogisticRegression,
            {
                "C": self.C,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            },
        )
        self._model.fit(x_np, y_np)
        model_classes = getattr(self._model, "classes_", None)
        self.classes_ = _to_numpy(model_classes) if model_classes is not None else np.unique(y_np)
        self.n_features_in_ = x_np.shape[1]
        if hasattr(self._model, "coef_"):
            self.coef_ = _to_numpy(getattr(self._model, "coef_"))
        return self

    def predict(self, x):
        """Prédit les labels en renvoyant un tableau numpy 1D."""
        if self._model is None:
            raise RuntimeError(NOT_FITTED_ERROR)
        return _to_numpy(self._model.predict(_dense_numpy(x)))


class CumlLinearSVCClassifier(ClassifierMixin, BaseEstimator):
    """Linear SVC GPU via cuML, compatible pipeline sklearn."""

    skip_cv = True
    is_gpu_model = True

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None

    def fit(self, x, y):
        """Entraîne le SVM linéaire GPU sur des features denses."""
        _require_cuml_deps()
        from cuml.svm import LinearSVC as CumlLinearSVC  # pyright: ignore[reportMissingImports]
        x_np = _dense_numpy(x)
        y_np = _normalize_labels(y)

        self._model = _construct_with_supported_kwargs(
            CumlLinearSVC,
            {
                "C": self.C,
                "max_iter": self.max_iter,
            },
        )
        self._model.fit(x_np, y_np)
        model_classes = getattr(self._model, "classes_", None)
        self.classes_ = _to_numpy(model_classes) if model_classes is not None else np.unique(y_np)
        self.n_features_in_ = x_np.shape[1]
        if hasattr(self._model, "coef_"):
            self.coef_ = _to_numpy(getattr(self._model, "coef_"))
        return self

    def predict(self, x):
        """Prédit les labels en renvoyant un tableau numpy 1D."""
        if self._model is None:
            raise RuntimeError(NOT_FITTED_ERROR)
        return _to_numpy(self._model.predict(_dense_numpy(x)))


class CumlKNNClassifier(ClassifierMixin, BaseEstimator):
    """KNN GPU via cuML, compatible pipeline sklearn."""

    skip_cv = True
    is_gpu_model = True

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform") -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._model = None

    def fit(self, x, y):
        """Entraîne le KNN GPU sur des features denses."""
        _require_cuml_deps()
        from cuml.neighbors import KNeighborsClassifier as CumlKNN  # pyright: ignore[reportMissingImports]
        x_np = _dense_numpy(x)
        y_np = _normalize_labels(y)
        effective_weights = self.weights
        if self.weights == "distance":
            # Certains environnements CUDA/NVRTC échouent sur le chemin
            # de pondération distance de cuML KNN (compilation dynamique CuPy).
            # On bascule en `uniform` pour préserver l'exécution des runs GPU.
            warnings.warn(
                "KNNGPU: weights='distance' est instable dans cet environnement CUDA; fallback vers 'uniform'.",
                RuntimeWarning,
            )
            effective_weights = "uniform"
        self.effective_weights_ = effective_weights

        self._model = _construct_with_supported_kwargs(
            CumlKNN,
            {
                "n_neighbors": self.n_neighbors,
                "weights": effective_weights,
            },
        )
        self._model.fit(x_np, y_np)
        model_classes = getattr(self._model, "classes_", None)
        self.classes_ = _to_numpy(model_classes) if model_classes is not None else np.unique(y_np)
        self.n_features_in_ = x_np.shape[1]
        return self

    def predict(self, x):
        """Prédit les labels en renvoyant un tableau numpy 1D."""
        if self._model is None:
            raise RuntimeError(NOT_FITTED_ERROR)
        return _to_numpy(self._model.predict(_dense_numpy(x)))


class CumlRandomForestClassifier(ClassifierMixin, BaseEstimator):
    """Random Forest GPU via cuML, compatible pipeline sklearn."""

    skip_cv = True
    is_gpu_model = True

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 24,
        max_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self._model = None

    def fit(self, x, y):
        """Entraîne le Random Forest GPU sur des features denses."""
        _require_cuml_deps()
        from cuml.ensemble import RandomForestClassifier as CumlRandomForest  # pyright: ignore[reportMissingImports]
        x_np = _dense_numpy(x)
        y_np = _normalize_labels(y)

        self._model = _construct_with_supported_kwargs(
            CumlRandomForest,
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features": self.max_features,
                "random_state": self.random_state,
            },
        )
        self._model.fit(x_np, y_np)
        model_classes = getattr(self._model, "classes_", None)
        self.classes_ = _to_numpy(model_classes) if model_classes is not None else np.unique(y_np)
        self.n_features_in_ = x_np.shape[1]
        if hasattr(self._model, "feature_importances_"):
            self.feature_importances_ = _to_numpy(getattr(self._model, "feature_importances_"))
        return self

    def predict(self, x):
        """Prédit les labels en renvoyant un tableau numpy 1D."""
        if self._model is None:
            raise RuntimeError(NOT_FITTED_ERROR)
        return _to_numpy(self._model.predict(_dense_numpy(x)))
