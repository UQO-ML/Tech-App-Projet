"""Définition et entraînement de plusieurs modèles de classification texte."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from utils import compute_metrics


def get_models(random_state: int = 42) -> dict[str, dict[str, Any]]:
    """Déclare les modèles classiques + leurs grilles d'hyperparamètres.

    Paramètres:
        random_state: Seed utilisée par les modèles stochastiques.

    Retour:
        Dictionnaire `{nom_modele: {"pipeline", "param_grid", "why"}}`.
    """
    return {
        "NaiveBayes": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("clf", MultinomialNB()),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__alpha": [0.5, 1.0],
            },
            "why": "Baseline rapide en texte; robuste avec des comptes/poids TF-IDF.",
        },
        "LogisticRegression": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__C": [0.5, 1.0, 2.0],
            },
            "why": "Très adaptée aux données textuelles clairsemées; souvent un excellent compromis performance/interprétabilité.",
        },
        "LinearSVC": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("clf", LinearSVC(class_weight="balanced", random_state=random_state)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "clf__C": [0.5, 1.0, 2.0],
            },
            "why": "SVM linéaire performant sur grands espaces de features TF-IDF et classes parfois déséquilibrées.",
        },
        "KNN": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=12000)),
                    ("clf", KNeighborsClassifier(n_neighbors=5)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__n_neighbors": [3, 5, 11],
                "clf__weights": ["uniform", "distance"],
            },
            "why": "Référence non paramétrique utile comme comparaison, même si souvent moins performante sur texte sparse.",
        },
        "DecisionTree": {
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=6000)),
                    ("clf", DecisionTreeClassifier(random_state=random_state, class_weight="balanced", ccp_alpha=0.0)),
                ],
                memory=None,
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__max_depth": [20, 40, None],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__ccp_alpha": [0.0, 0.001],
            },
            "why": "Modèle interprétable pour analyser des règles décisionnelles, pertinent comme repère explicatif.",
        },
        "RandomForest": {
            "pipeline": Pipeline(
                [
                    # On limite les features pour garder un temps de calcul raisonnable.
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
            ),
            "param_grid": {
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2, 5],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 30],
                "clf__max_features": ["sqrt", "log2"],
                "clf__min_samples_leaf": [1, 2],
            },
            "why": "Ensemble d'arbres plus stable qu'un arbre simple; capture des interactions non linéaires.",
        },
    }


def _distilbert_deps_available() -> bool:
    """Vérifie si les dépendances deep learning nécessaires à DistilBERT sont présentes."""
    try:
        import datasets  # noqa: F401  # pyright: ignore[reportMissingImports]
        import torch  # noqa: F401
        import transformers  # noqa: F401  # pyright: ignore[reportMissingImports]

        return True
    except Exception:
        return False


def _set_random_seeds(random_state: int) -> None:
    """Fixe les seeds Python/Numpy/PyTorch pour améliorer la reproductibilité."""
    import torch

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)


def _build_failed_result(error_message: str) -> dict[str, Any]:
    """Construit une structure de résultat standard en cas d'échec."""
    return {
        "status": "failed",
        "estimator": None,
        "val_metrics": {},
        "tuning": {},
        "error": error_message,
    }


def _build_trained_result(estimator: Any, y_val, y_val_pred, tuning: dict[str, Any]) -> dict[str, Any]:
    """Construit une structure de résultat standard après entraînement réussi."""
    return {
        "status": "trained",
        "estimator": estimator,
        "val_metrics": compute_metrics(y_val, y_val_pred),
        "tuning": tuning,
        "error": None,
    }


def get_expected_model_names(include_distilbert: bool = True) -> list[str]:
    """Retourne la liste des modèles qui doivent apparaître dans le report final."""
    names = list(get_models().keys())
    if include_distilbert:
        names.append("DistilBERT")
    return names


class DistilBertTextClassifier:
    """
    Classifieur DistilBERT avec API proche sklearn (fit/predict).
    Entraînement compact pour rester compatible avec un pipeline de cours.
    """

    skip_cv = True
    is_deep_model = True

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.label_to_id: dict[int, int] = {}
        self.id_to_label: dict[int, int] = {}
        self._trainer = None
        self._tokenizer = None
        self._is_fitted = False

    def _build_label_mapping(self, y_train) -> None:
        """Construit la correspondance labels originaux <-> ids attendus par Hugging Face."""
        labels = sorted({int(label) for label in y_train})
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """Fine-tune DistilBERT sur train (et val si fournie)."""
        from datasets import Dataset  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        _set_random_seeds(self.random_state)
        self._build_label_mapping(y_train)

        train_texts = [str(text) for text in x_train]
        train_labels = [self.label_to_id[int(label)] for label in y_train]

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        eval_dataset = None
        if x_val is not None and y_val is not None:
            val_texts = [str(text) for text in x_val]
            val_labels = [self.label_to_id[int(label)] for label in y_val]
            eval_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_to_id),
        )

        def tokenize(batch):
            """Tokenise un batch de textes pour DistilBERT."""
            return self._tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_length,
            )

        train_dataset = train_dataset.map(tokenize, batched=True)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(tokenize, batched=True)

        output_dir = Path(__file__).resolve().parent.parent / "Outputs" / "models" / "distilbert_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        base_training_args = {
            "output_dir": str(output_dir),
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "num_train_epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "save_strategy": "no",
            "logging_strategy": "epoch",
            "report_to": "none",
            "seed": self.random_state,
        }

        # Compatibilité transformers: certaines versions attendent eval_strategy, d'autres evaluation_strategy.
        try:
            training_args = TrainingArguments(
                eval_strategy="epoch" if eval_dataset is not None else "no",
                **base_training_args,
            )
        except TypeError:
            training_args = TrainingArguments(
                evaluation_strategy="epoch" if eval_dataset is not None else "no",
                **base_training_args,
            )

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": DataCollatorWithPadding(tokenizer=self._tokenizer),
        }
        # Compatibilité transformers: tokenizer (ancien) vs processing_class (récent).
        try:
            self._trainer = Trainer(processing_class=self._tokenizer, **trainer_kwargs)
        except TypeError:
            self._trainer = Trainer(tokenizer=self._tokenizer, **trainer_kwargs)
        self._trainer.train()
        self._is_fitted = True
        return self

    def predict(self, x):
        """Prédit des labels de classes originales (0/1/2)."""
        if not self._is_fitted or self._trainer is None or self._tokenizer is None:
            raise RuntimeError("Le modèle DistilBERT doit être entraîné avant predict().")

        from datasets import Dataset  # pyright: ignore[reportMissingImports]

        texts = [str(text) for text in x]
        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch):
            """Tokenise un batch de textes pour l'inférence DistilBERT."""
            return self._tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_length,
            )

        dataset = dataset.map(tokenize, batched=True)
        predictions = self._trainer.predict(dataset).predictions
        pred_ids = predictions.argmax(axis=1)
        return np.array([self.id_to_label[int(idx)] for idx in pred_ids])


def train_with_grid_search(
    x_train,
    y_train,
    model_name: str,
    random_state: int = 42,
    cv: int = 5,
    scoring: str = "f1_macro",
) -> tuple[Any, dict[str, Any]]:
    """Optimise un modèle sklearn via GridSearchCV.

    Paramètres:
        x_train, y_train: Données d'entraînement.
        model_name: Nom du modèle déclaré dans `get_models`.
        random_state: Seed de reproductibilité.
        cv: Nombre de folds pour la validation croisée.
        scoring: Métrique optimisée par GridSearch.

    Retour:
        Tuple `(best_estimator, tuning_info)`.
    """
    model_entry = get_models(random_state=random_state)[model_name]
    if not model_entry["param_grid"]:
        estimator = model_entry["pipeline"]
        estimator.fit(x_train, y_train)
        return estimator, {"best_params": {}, "best_cv_score": None}

    search = GridSearchCV(
        estimator=model_entry["pipeline"],
        param_grid=model_entry["param_grid"],
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
    }


def _train_single_classic_model(
    model_name: str,
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int,
) -> dict[str, Any]:
    """Entraîne un modèle classique et retourne un résultat normalisé."""
    estimator, tuning_info = train_with_grid_search(
        x_train,
        y_train,
        model_name=model_name,
        random_state=random_state,
    )
    y_val_pred = estimator.predict(x_val)
    return _build_trained_result(estimator=estimator, y_val=y_val, y_val_pred=y_val_pred, tuning=tuning_info)


def train_distilbert(
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int = 42,
    distilbert_epochs: int = 1,
) -> dict[str, Any] | None:
    """Entraîne DistilBERT en fine-tuning direct.

    Retour:
        Un dictionnaire compatible avec le format de résultats global, ou `None` si indisponible.
    """
    if not _distilbert_deps_available():
        return None

    estimator = DistilBertTextClassifier(
        random_state=random_state,
        epochs=distilbert_epochs,
        batch_size=16,
        max_length=128,
    )
    estimator.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    y_val_pred = estimator.predict(x_val)
    return {
        "estimator": estimator,
        "val_metrics": compute_metrics(y_val, y_val_pred),
        "tuning": {
            "best_params": {
                "model_name": estimator.model_name,
                "epochs": estimator.epochs,
                "batch_size": estimator.batch_size,
                "max_length": estimator.max_length,
                "learning_rate": estimator.learning_rate,
            },
            "best_cv_score": None,
            "note": "DistilBERT entraîné via fine-tuning direct (pas de GridSearchCV pour limiter le coût).",
        },
    }


def _train_or_skip_distilbert(
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int,
    distilbert_epochs: int,
) -> dict[str, Any]:
    """Entraîne DistilBERT, sinon retourne un statut `skipped`/`failed` explicite."""
    if not _distilbert_deps_available():
        return {
            "status": "skipped",
            "estimator": None,
            "val_metrics": {},
            "tuning": {},
            "error": "Dépendances manquantes: torch/transformers/datasets.",
        }
    distilbert_result = train_distilbert(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_state=random_state,
        distilbert_epochs=distilbert_epochs,
    )
    if distilbert_result is None:
        return _build_failed_result("DistilBERT indisponible malgré dépendances déclarées.")
    distilbert_result["status"] = "trained"
    distilbert_result["error"] = None
    return distilbert_result


def train_all_models(
    x_train,
    y_train,
    x_val,
    y_val,
    random_state: int = 42,
    include_distilbert: bool = True,
    distilbert_epochs: int = 1,
) -> dict[str, dict[str, Any]]:
    """Entraîne tous les modèles demandés (classiques + DistilBERT optionnel).

    Retour:
        Dictionnaire des résultats par modèle avec statut, métriques val, tuning et erreur éventuelle.
    """
    results: dict[str, dict[str, Any]] = {}
    for model_name in get_models(random_state=random_state):
        try:
            results[model_name] = _train_single_classic_model(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                random_state=random_state,
            )
        except Exception as exc:
            results[model_name] = _build_failed_result(str(exc))

    if include_distilbert:
        try:
            results["DistilBERT"] = _train_or_skip_distilbert(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                random_state=random_state,
                distilbert_epochs=distilbert_epochs,
            )
        except Exception as exc:
            results["DistilBERT"] = _build_failed_result(str(exc))
    return results


def cross_validate_estimator(estimator, X, y, cv: int = 5) -> list[float]:
    """Retourne les scores F1 macro en validation croisée.

    Notes:
        - Si `estimator.skip_cv` est vrai (cas DistilBERT), renvoie une liste vide.
    """
    if getattr(estimator, "skip_cv", False):
        return []
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return [float(score) for score in scores]


def get_model_rationales(random_state: int = 42) -> dict[str, str]:
    """Retourne un texte court de justification pour chaque modèle."""
    rationales = {name: entry.get("why", "") for name, entry in get_models(random_state=random_state).items()}
    rationales["DistilBERT"] = (
        "Modèle Transformer pré-entraîné qui capte mieux le contexte sémantique des tweets; "
        "souvent plus performant sur la détection de nuances offensantes/haineuses."
    )
    return rationales
