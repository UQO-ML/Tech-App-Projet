"""Définition du modèle DistilBERT et helpers associés."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler


def distilbert_deps_available() -> bool:
    """Vérifie si les dépendances DistilBERT sont disponibles."""
    try:
        import datasets  # noqa: F401  # pyright: ignore[reportMissingImports]
        import torch  # noqa: F401
        import transformers  # noqa: F401  # pyright: ignore[reportMissingImports]


        return True
    except Exception:
        return False

from datasets import Dataset
from transformers import (  # pyright: ignore[reportMissingImports]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

def _set_random_seeds(random_state: int) -> None:
    """Fixe les seeds Python/Numpy/PyTorch."""

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

class DistilBertTextClassifier(ClassifierMixin, BaseEstimator):
    """Classifieur DistilBERT compatible avec les conventions sklearn."""

    skip_cv = False
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

        texts = [str(text) for text in x]
        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch):
            """Tokenise un batch pour l'inférence."""
            return self._tokenizer(batch["text"], truncation=True, max_length=self.max_length)

        dataset = dataset.map(tokenize, batched=True)
        predictions = self._trainer.predict(dataset).predictions
        pred_ids = predictions.argmax(axis=1)
        return np.array([self.id_to_label[int(idx)] for idx in pred_ids])

    def predict_proba(self, x):
        """Retourne les probabilités softmax par classe dans l'ordre des labels originaux."""
        if not self._is_fitted or self._trainer is None or self._tokenizer is None:
            raise RuntimeError("Le modèle DistilBERT doit être entraîné avant predict_proba().")

        dataset = self._tokenize_ds(x)
        logits = self._trainer.predict(dataset).predictions
        probs_internal = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()

        # Remap ordre interne HF -> ordre labels originaux [0,1,2]
        n_classes = len(self.label_to_id)
        probs_out = np.zeros((probs_internal.shape[0], n_classes), dtype=float)
        for internal_idx, orig_label in self.id_to_label.items():
            probs_out[:, int(orig_label)] = probs_internal[:, int(internal_idx)]
        return probs_out

    def save_local(self, export_dir: str | Path) -> None:
        if not self._is_fitted or self._trainer is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted.")
        export = Path(export_dir)
        export.mkdir(parents=True, exist_ok=True)

        self._trainer.model.save_pretrained(export / "hf_model")
        self._tokenizer.save_pretrained(export / "hf_model")

        meta = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
            "hate_threshold": getattr(self, "hate_threshold", 0.5),
        }
        (export / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load_local(cls, export_dir: str | Path):

        export = Path(export_dir)
        meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))

        obj = cls(
            model_name=meta["model_name"],
            max_length=meta["max_length"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            learning_rate=meta["learning_rate"],
            weight_decay=meta["weight_decay"],
            random_state=meta["random_state"],
        )
        obj.label_to_id = {int(k): int(v) for k, v in meta["label_to_id"].items()}
        obj.id_to_label = {int(k): int(v) for k, v in meta["id_to_label"].items()}
        obj.hate_threshold = float(meta.get("hate_threshold", 0.5))

        obj._tokenizer = AutoTokenizer.from_pretrained(export / "hf_model")
        model = AutoModelForSequenceClassification.from_pretrained(export / "hf_model")
        args = TrainingArguments(output_dir=str(export / "tmp"), report_to="none")
        obj._trainer = Trainer(model=model, args=args, tokenizer=obj._tokenizer)
        obj._is_fitted = True
        return obj


class WeightedFocalSamplerTrainer:
    """Factory helper for a Trainer subclass with weighted/focal loss and optional balanced sampler."""

    @staticmethod
    def build(base_trainer_cls):
        class _WeightedFocalSamplerTrainer(base_trainer_cls):
            def __init__(
                self,
                *args,
                class_weights=None,
                focal_gamma: float = 0.0,
                train_label_ids: list[int] | None = None,
                use_balanced_sampler: bool = True,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
                self.focal_gamma = float(focal_gamma)
                self.train_label_ids = train_label_ids
                self.use_balanced_sampler = bool(use_balanced_sampler)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                cw = self.class_weights.to(logits.device) if self.class_weights is not None else None

                if self.focal_gamma > 0:
                    probs = torch.softmax(logits, dim=-1)
                    p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
                    focal_factor = (1.0 - p_t) ** self.focal_gamma
                    per_sample_ce = F.cross_entropy(logits, labels, weight=cw, reduction="none")
                    loss = (focal_factor * per_sample_ce).mean()
                else:
                    loss = F.cross_entropy(logits, labels, weight=cw)

                return (loss, outputs) if return_outputs else loss

            def get_train_dataloader(self):
                if not self.use_balanced_sampler or self.train_label_ids is None:
                    return super().get_train_dataloader()

                # Keep only model-forward columns (input_ids, attention_mask, labels, ...)
                # so DataCollatorWithPadding doesn't try to tensorize raw string columns like `text`.
                train_dataset = self._remove_unused_columns(self.train_dataset, description="training")
                data_collator = self._get_collator_with_removed_columns(self.data_collator, description="training")

                labels = np.asarray(self.train_label_ids)
                counts = np.bincount(labels)
                inv = np.zeros_like(counts, dtype=float)
                non_zero = counts > 0
                inv[non_zero] = 1.0 / counts[non_zero]
                sample_weights = inv[labels]

                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )

                return DataLoader(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=sampler,
                    collate_fn=data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )

        return _WeightedFocalSamplerTrainer


class OptimizedDistilBertClassifier(ClassifierMixin, BaseEstimator):
    """DistilBERT with weighted/focal loss, balanced sampler, warmup, early stopping and threshold tuning."""

    skip_cv = False
    is_deep_model = True

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 160,
        epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        random_state: int = 42,
        warmup_ratio: float = 0.1,
        focal_gamma: float = 0.0,
        use_balanced_sampler: bool = True,
        enable_error_driven_finetune: bool = False,
        error_driven_repeat: int = 2,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.warmup_ratio = float(warmup_ratio)
        self.focal_gamma = float(focal_gamma)
        self.use_balanced_sampler = bool(use_balanced_sampler)
        self.enable_error_driven_finetune = bool(enable_error_driven_finetune)
        self.error_driven_repeat = int(error_driven_repeat)

        self.label_to_id: dict[int, int] = {}
        self.id_to_label: dict[int, int] = {}
        self._tokenizer = None
        self._trainer = None
        self._is_fitted = False
        self.hate_threshold = 0.5
        self.threshold_metrics: dict[str, float | None] = {}

    def _build_mappings(self, y_train) -> None:
        labels = sorted({int(label) for label in y_train})
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def _tokenize_ds(self, texts, labels=None):

        payload = {"text": [str(text) for text in texts]}
        if labels is not None:
            payload["labels"] = [int(value) for value in labels]
        dataset = Dataset.from_dict(payload)
        tokenized = dataset.map(
            lambda batch: self._tokenizer(batch["text"], truncation=True, max_length=self.max_length),
            batched=True,
        )
        # Keep only numeric/token columns for HF collator.
        return tokenized.remove_columns(["text"]) if "text" in tokenized.column_names else tokenized

    def _compute_metrics_builder(self, hate_label: int = 0):
        def _compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            y_true = np.array([self.id_to_label[int(idx)] for idx in labels])
            y_pred = np.array([self.id_to_label[int(idx)] for idx in preds])
            f1_macro = f1_score(y_true, y_pred, average="macro")
            _, recall_macro, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            hate_recall = precision_recall_fscore_support(
                y_true, y_pred, labels=[hate_label], average=None, zero_division=0
            )[1][0]
            return {
                "f1_macro": float(f1_macro),
                "recall_macro": float(recall_macro),
                "hate_recall": float(hate_recall),
            }

        return _compute

    def _tune_hate_threshold(self, logits, y_true_orig, hate_label: int = 0):
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        hate_internal_id = next((key for key, value in self.id_to_label.items() if value == hate_label), None)
        if hate_internal_id is None:
            return 0.5, {"best_f1_macro": None, "best_hate_recall": None}

        base_pred = probs.argmax(axis=1)
        best_t, best_f1, best_hrec = 0.5, -1.0, -1.0
        for threshold in np.linspace(0.20, 0.80, 31):
            pred = base_pred.copy()
            pred[probs[:, hate_internal_id] >= threshold] = hate_internal_id
            y_pred_orig = np.array([self.id_to_label[int(idx)] for idx in pred])
            f1_macro = f1_score(y_true_orig, y_pred_orig, average="macro")
            hate_recall = precision_recall_fscore_support(
                y_true_orig, y_pred_orig, labels=[hate_label], average=None, zero_division=0
            )[1][0]
            if (f1_macro > best_f1) or (np.isclose(f1_macro, best_f1) and hate_recall > best_hrec):
                best_t, best_f1, best_hrec = float(threshold), float(f1_macro), float(hate_recall)

        return best_t, {"best_f1_macro": best_f1, "best_hate_recall": best_hrec}

    def fit(self, x_train, y_train, x_val=None, y_val=None):

        set_seed(self.random_state)
        self._build_mappings(y_train)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.label_to_id))

        y_train_id = [self.label_to_id[int(label)] for label in y_train]
        train_dataset = self._tokenize_ds(x_train, y_train_id)

        eval_dataset = None
        y_val_id = None
        if x_val is not None and y_val is not None:
            y_val_id = [self.label_to_id[int(label)] for label in y_val]
            eval_dataset = self._tokenize_ds(x_val, y_val_id)

        classes = np.unique(y_train_id)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(y_train_id))
        class_weights = torch.tensor(weights, dtype=torch.float32)

        training_args = TrainingArguments(
            output_dir="./distilbert_ckpt",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch" if eval_dataset is not None else "no",
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to="none",
            seed=self.random_state,
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset is not None else []
        trainer_cls = WeightedFocalSamplerTrainer.build(Trainer)
        self._trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self._tokenizer),
            compute_metrics=self._compute_metrics_builder(hate_label=0) if eval_dataset is not None else None,
            class_weights=class_weights,
            focal_gamma=self.focal_gamma,
            train_label_ids=y_train_id,
            use_balanced_sampler=self.use_balanced_sampler,
            callbacks=callbacks,
        )

        self._trainer.train()

        if eval_dataset is not None:
            val_pred = self._trainer.predict(eval_dataset)
            y_val_orig = np.array([self.id_to_label[int(idx)] for idx in y_val_id])
            self.hate_threshold, self.threshold_metrics = self._tune_hate_threshold(
                val_pred.predictions, y_val_orig, hate_label=0
            )

            if self.enable_error_driven_finetune:
                logits = val_pred.predictions
                pred_ids = logits.argmax(axis=1)
                y_pred_orig = np.array([self.id_to_label[int(idx)] for idx in pred_ids])
                false_negative_hate_mask = (np.array(y_val, dtype=int) == 0) & (y_pred_orig != 0)
                hard_texts = list(np.array([str(text) for text in x_val])[false_negative_hate_mask])
                hard_labels = list(np.array(y_val, dtype=int)[false_negative_hate_mask])
                if hard_texts:
                    augmented_texts = [str(text) for text in x_train] + hard_texts * self.error_driven_repeat
                    augmented_labels = [int(label) for label in y_train] + hard_labels * self.error_driven_repeat
                    augmented_ids = [self.label_to_id[int(label)] for label in augmented_labels]
                    augmented_dataset = self._tokenize_ds(augmented_texts, augmented_ids)
                    self._trainer.train_dataset = augmented_dataset
                    self._trainer.train_label_ids = augmented_ids
                    self._trainer.train()

        self._is_fitted = True
        return self

    def predict(self, x):
        if not self._is_fitted or self._trainer is None or self._tokenizer is None:
            raise RuntimeError("Le modèle DistilBERT doit être entraîné avant predict().")

        dataset = self._tokenize_ds(x)
        logits = self._trainer.predict(dataset).predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_ids = probs.argmax(axis=1)

        hate_internal_id = next((key for key, value in self.id_to_label.items() if value == 0), None)
        if hate_internal_id is not None:
            pred_ids[probs[:, hate_internal_id] >= self.hate_threshold] = hate_internal_id

        return np.array([self.id_to_label[int(idx)] for idx in pred_ids])

    def predict_proba(self, x):
        """Retourne les probabilités softmax par classe dans l'ordre des labels originaux."""
        if not self._is_fitted or self._trainer is None or self._tokenizer is None:
            raise RuntimeError("Le modèle DistilBERT doit être entraîné avant predict_proba().")

        dataset = self._tokenize_ds(x)
        logits = self._trainer.predict(dataset).predictions
        probs_internal = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()

        # Remap ordre interne HF -> ordre labels originaux [0,1,2]
        n_classes = len(self.label_to_id)
        probs_out = np.zeros((probs_internal.shape[0], n_classes), dtype=float)
        for internal_idx, orig_label in self.id_to_label.items():
            probs_out[:, int(orig_label)] = probs_internal[:, int(internal_idx)]
        return probs_out


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


