"""
evaluate.py — Evaluation, metrics, and reporting (INF 6243).

Role:
  - Load a trained model and test/validation data.
  - Run inference, compute metrics (accuracy, F1, confusion matrix, etc.) and optionally per-class metrics.
  - Print results and/or save to file (CSV, JSON, or report in OUTPUT_DIR).

Structure:
  1. Imports: torch, numpy, sklearn.metrics (accuracy_score, f1_score, classification_report, confusion_matrix), config (get_device, paths).
  2. load_model(path, device): load checkpoint and model; return model in eval mode.
  3. predict(model, loader, device): run model on loader; return predictions and true labels (or probs).
  4. compute_metrics(y_true, y_pred): return dict with accuracy, f1, etc., and optionally confusion matrix.
  5. run(config, model_path, test_loader):
     - model = load_model(model_path, device)
     - y_true, y_pred = predict(model, test_loader, device)
     - metrics = compute_metrics(y_true, y_pred)
     - print or save report; return metrics
  6. Optional: ROC/AUC, per-class precision/recall, export predictions to CSV.
"""

# Placeholder: implement evaluation and metrics
# from config import get_device, OUTPUT_DIR


def run(config=None, model_path=None, test_loader=None):
    """
    Evaluate model on test_loader. config: device, output_dir. model_path: path to checkpoint.
    Returns dict of metrics.
    """
    # Implement: load model -> predict -> compute_metrics -> report
    return {}
