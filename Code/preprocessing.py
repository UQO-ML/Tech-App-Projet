"""Prétraitement et résumé EDA pour la classification de tweets haineux/offensants."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_LABELS = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}


def load_data(path_or_url: str | Path) -> pd.DataFrame:
    """Charge un CSV local ou distant et retourne un DataFrame."""
    return pd.read_csv(path_or_url, encoding="utf-8")


def clean_text(text: str) -> str:
    """Nettoyage minimal: minuscules, suppression URL/mentions/symboles, espaces propres."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les lignes invalides puis crée des variables utiles à l'EDA."""
    clean_df = df.copy()

    # Supprime les colonnes inutiles fréquentes dans le dataset Kaggle.
    to_drop = [col for col in ["Unnamed: 0", "Unnamed: 0.1"] if col in clean_df.columns]
    if to_drop:
        clean_df = clean_df.drop(columns=to_drop)

    clean_df = clean_df.dropna(subset=["tweet", "class"]).drop_duplicates(subset=["tweet", "class"])
    clean_df["class"] = clean_df["class"].astype(int)
    clean_df["clean_tweet"] = clean_df["tweet"].map(clean_text)
    clean_df["tweet_length"] = clean_df["tweet"].astype(str).str.len()
    clean_df["word_count"] = clean_df["clean_tweet"].str.split().str.len()
    return clean_df.reset_index(drop=True)


def exploratory_summary(df: pd.DataFrame, target_column: str = "class") -> dict[str, Any]:
    """Retourne un résumé EDA exploitable dans le notebook et le rapport."""
    numeric_df = df.select_dtypes(include=["number"])
    summary = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "describe": df.describe(include="all").fillna("").to_dict(),
        "class_distribution": df[target_column].value_counts().sort_index().to_dict(),
        "class_distribution_pct": (df[target_column].value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
        "correlation": numeric_df.corr(numeric_only=True).fillna(0).round(3).to_dict() if not numeric_df.empty else {},
    }
    return summary


def train_val_test_split(
    x: pd.Series,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split stratifié: train/val/test."""
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
