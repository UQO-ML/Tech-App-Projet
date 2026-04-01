"""Définition du modèle DistilBERT et helpers associés."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def distilbert_deps_available() -> bool:
    """Vérifie si les dépendances DistilBERT sont disponibles."""
    try:
        import datasets  # noqa: F401  # pyright: ignore[reportMissingImports]
        import torch  # noqa: F401
        import transformers  # noqa: F401  # pyright: ignore[reportMissingImports]

        return True
    except Exception:
        return False


def _set_random_seeds(random_state: int) -> None:
    """Fixe les seeds Python/Numpy/PyTorch."""
    import torch

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)


class DistilBertTextClassifier(ClassifierMixin, BaseEstimator):
    """Classifieur DistilBERT compatible avec les conventions sklearn."""

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
        """Construit la correspondance labels originaux <-> ids HF."""
        labels = sorted({int(label) for label in y_train})
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """Fine-tune DistilBERT sur les données d'entraînement."""
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
            """Tokenise un batch pour DistilBERT."""
            return self._tokenizer(batch["text"], truncation=True, max_length=self.max_length)

        train_dataset = train_dataset.map(tokenize, batched=True)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(tokenize, batched=True)

        output_dir = Path(__file__).resolve().parents[2] / "Outputs" / "models" / "distilbert_run"
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

        # Compatibilité de noms d'arguments entre versions de transformers.
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

        # Compatibilité tokenizer/processing_class entre versions.
        try:
            self._trainer = Trainer(processing_class=self._tokenizer, **trainer_kwargs)
        except TypeError:
            self._trainer = Trainer(tokenizer=self._tokenizer, **trainer_kwargs)

        self._trainer.train()
        self._is_fitted = True
        return self

    def predict(self, x):
        """Prédit des labels dans l'espace original (0/1/2)."""
        if not self._is_fitted or self._trainer is None or self._tokenizer is None:
            raise RuntimeError("Le modèle DistilBERT doit être entraîné avant predict().")

        from datasets import Dataset  # pyright: ignore[reportMissingImports]

        texts = [str(text) for text in x]
        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch):
            """Tokenise un batch pour l'inférence."""
            return self._tokenizer(batch["text"], truncation=True, max_length=self.max_length)

        dataset = dataset.map(tokenize, batched=True)
        predictions = self._trainer.predict(dataset).predictions
        pred_ids = predictions.argmax(axis=1)
        return np.array([self.id_to_label[int(idx)] for idx in pred_ids])


def build_distilbert_tuning(estimator: DistilBertTextClassifier) -> dict[str, Any]:
    """Retourne les hyperparamètres pertinents de DistilBERT."""
    return {
        "best_params": {
            "model_name": estimator.model_name,
            "epochs": estimator.epochs,
            "batch_size": estimator.batch_size,
            "max_length": estimator.max_length,
            "learning_rate": estimator.learning_rate,
        },
        "best_cv_score": None,
        "note": "DistilBERT entraîné via fine-tuning direct (pas de GridSearchCV pour limiter le coût).",
    }
