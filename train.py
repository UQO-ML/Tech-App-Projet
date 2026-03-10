"""
train.py — Training loop and model fitting (INF 6243).

Role:
  - Build or load a model (from config or checkpoint).
  - Run the training loop: iterate over train loader, forward, loss, backward, optimizer step.
  - Optionally validate each epoch and save best checkpoint (by loss or metric).
  - Log progress (epoch, train loss, val loss/metric) via print or a logger.

Structure:
  1. Imports: torch, tqdm, config (get_device, paths, hyperparams), data (get_dataloaders), model definition if separate.
  2. build_model(config, device): instantiate model, move to device.
  3. train_epoch(model, loader, optimizer, criterion, device): one epoch loop; return mean loss.
  4. validate(model, loader, criterion, device): optional; return loss and/or accuracy.
  5. run(config, train_loader, val_loader=None):
     - model = build_model(config, device)
     - optimizer = Adam/SGD(model.parameters(), lr=config.learning_rate)
     - criterion = CrossEntropyLoss or assignment-specific
     - for epoch in range(config.epochs): train_epoch; validate; save_best; log
  6. Optional: checkpoint saving (path from config.MODEL_DIR), early stopping, LR scheduler.
"""

# Placeholder: implement training loop
# from config import get_device, EPOCHS, LEARNING_RATE, MODEL_DIR


def run(config=None, train_loader=None, val_loader=None):
    """
    Run full training. config: object with device, epochs, lr, model_dir, etc.
    train_loader, val_loader: from data.get_dataloaders().
    """
    # Implement: build model -> optimizer -> for each epoch: train_epoch, validate, save best
    pass
