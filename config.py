"""
config.py — Central configuration for Tech-App-Devoir-II (INF 6243).

Role:
  - Define default paths (data dir, model save dir, outputs).
  - Define hyperparameters (batch size, epochs, learning rate, etc.).
  - Expose device selection (CPU vs CUDA) so the rest of the code uses a single place.

Structure:
  1. Imports: os, optional torch for device.
  2. Paths: DATA_DIR, MODEL_DIR, OUTPUT_DIR (or from env vars for portability).
  3. Data: train/val/test split ratios, batch size, num workers.
  4. Training: epochs, learning rate, optimizer name, weight decay, etc.
  5. Model: architecture name or key, input size, num classes.
  6. Device: get_device() -> torch.device("cuda" if torch.cuda.is_available() else "cpu").
  7. Optional: from_dict() / to_dict() or dataclass for overrides from CLI or YAML.
"""

# Paths (adjust or override via environment variables)
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# Data
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 on Windows often avoids multiprocessing issues; increase on Linux if needed

# Training (placeholders)
EPOCHS = 10
LEARNING_RATE = 1e-3

# Device: set by get_device() so train/evaluate use CUDA when available
def get_device():
    """Return torch.device('cuda' or 'cpu'). Implement using torch.cuda.is_available()."""
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return None
