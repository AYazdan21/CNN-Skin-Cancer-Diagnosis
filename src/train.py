"""
src/train.py
Training pipeline with Adam optimizer, CrossEntropyLoss, and Callbacks:
- Multi-GPU support via nn.DataParallel (e.g. Kaggle Dual T4 GPUs)
- Model Checkpoint (saves unwrapped weights based on val_loss)
- ReduceLROnPlateau (halves LR when val_loss plateaus)
- Early Stopping (stops after patience epochs without improvement)
- Learning curves visualization (plots Loss and Accuracy over epochs like Fig. 5 & 6)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config import (
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
    DEVICE,
)


class EarlyStopping:
    """
    Early stops training if validation loss doesn't improve after a given patience.
    Also manages saving the best model checkpoint.
    """

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, delta: float = 1e-4, checkpoint_path: Path = None):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.checkpoint_path:
                # Unwrap model if wrapped in DataParallel
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_to_save.state_dict(),
                        "val_loss": val_loss,
                    },
                    self.checkpoint_path,
                )
                print(f"[*] Checkpoint saved at epoch {epoch}: val_loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"[!] EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model: nn.Module, dataloader, criterion, optimizer, device):
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model: nn.Module, dataloader, criterion, device):
    """Evaluates the model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def plot_training_history(history: dict, save_path: Path):
    """
    Plots training vs validation loss and accuracy curves (Paper Fig. 5 & 6).
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs_range, history["train_loss"], label="Training Loss", color="#1f77b4")
    ax1.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#ff7f0e")
    ax1.set_title("Training and Validation Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs_range, history["train_acc"], label="Training Accuracy", color="#2ca02c")
    ax2.plot(epochs_range, history["val_acc"], label="Validation Accuracy", color="#d62728")
    ax2.set_title("Training and Validation Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Training curves saved to {save_path}")


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    class_weights: torch.Tensor = None,
    device: torch.device = DEVICE,
    checkpoint_dir: Path = CHECKPOINTS_DIR,
    output_dir: Path = OUTPUTS_DIR,
):
    """
    Main training function orchestrating:
    - Multi-GPU DataParallel wrapping if available
    - Adam optimizer
    - CrossEntropyLoss (with optional class weights)
    - ReduceLROnPlateau scheduler
    - EarlyStopping & Checkpointing
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    model = model.to(device)

    # Support Multi-GPU (e.g. Dual T4 GPUs on Kaggle)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[*] Multi-GPU: Detected {torch.cuda.device_count()} GPUs. Using nn.DataParallel!")
        model = nn.DataParallel(model)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"[*] Applied class weights to CrossEntropyLoss: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
    )
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=best_checkpoint_path,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    print(f"\n================ Starting Training ================")
    print(f"Device: {device} (Count: {torch.cuda.device_count()}) | Max Epochs: {epochs} | Initial LR: {lr}")
    print(f"Checkpoints directory: {best_checkpoint_path}")
    print(f"===================================================\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% "
            f"| LR: {current_lr:.2e}"
        )

        # Evaluate Early Stopping and Checkpoint Save
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"\n[!] Early stopping triggered at epoch {epoch}. Stopping training.")
            break

    # Save training history JSON
    history_file = output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[*] Training history saved to {history_file}")

    # Plot training curves
    plot_training_history(history, output_dir / "training_curves.png")

    print(f"\nTraining completed! Best checkpoint from epoch {early_stopping.best_epoch} saved to {best_checkpoint_path}")
    return history
