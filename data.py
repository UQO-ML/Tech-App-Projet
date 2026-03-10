"""
data.py — Data loading, preprocessing, and dataset utilities (INF 6243).

Role:
  - Load raw data (CSV, images, or other) and build PyTorch Dataset(s) or sklearn-friendly arrays.
  - Apply preprocessing (normalization, tokenization, train/val/test splits).
  - Expose DataLoader(s) or (X_train, y_train, X_val, y_val, ...) for train.py and evaluate.py.

Structure:
  1. Imports: pathlib, numpy, pandas, optional torch (Dataset, DataLoader), optional sklearn (train_test_split).
  2. Constants or config: paths from config.py, column names, image size, etc.
  3. Preprocessing functions: e.g. normalize(), transform_targets(), clean_text().
  4. Dataset class(es): subclass torch.utils.data.Dataset if using PyTorch; __len__, __getitem__.
  5. get_dataloaders(config) or get_train_val_test(config):
     - Build Dataset(s), apply transforms.
     - Return DataLoader(s) or (X_train, y_train, X_val, y_val, X_test, y_test).
  6. Optional: save/load preprocessed data to disk to avoid recomputing.
"""

# Placeholder: implement dataset construction and loaders
# from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS


def get_dataloaders(config=None):
    """
    Build and return train/val (and optionally test) DataLoaders or arrays.
    config: object or dict with DATA_DIR, BATCH_SIZE, NUM_WORKERS, etc.
    """
    if config is None:
        config = type("Config", (), {"batch_size": 32, "num_workers": 0})()
    # Implement: load data -> Dataset(s) -> DataLoader(s)
    return None, None
