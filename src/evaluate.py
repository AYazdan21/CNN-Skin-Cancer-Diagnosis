"""
src/evaluate.py
Evaluation module reproducing all metrics from the paper:
- Test Accuracy, Precision, Recall, and F1-score (Table 1, 4, Fig. 8)
- Confusion Matrix visualization (Fig. 9)
- Multi-class ROC Curve and AUC (Fig. 10)
- Regression metrics: MSE, RMSE, MAE (Fig. 11, Eq. 24-26)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
)
import torch
import torch.nn.functional as F

from src.config import (
    CLASS_MAPPING,
    CLASS_NAMES,
    NUM_CLASSES,
    OUTPUTS_DIR,
    CHECKPOINTS_DIR,
    DEVICE,
)
from src.model import build_model


def run_inference(model: torch.nn.Module, dataloader, device: torch.device):
    """
    Runs model on dataloader and returns:
    - true_labels: 1D numpy array
    - pred_labels: 1D numpy array
    - pred_probs: 2D numpy array [N, num_classes] (softmax probabilities)
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """Plots and saves the confusion matrix heatmap (Paper Fig. 9)."""
    short_names = list(CLASS_MAPPING.keys())
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
    )
    plt.title("Confusion Matrix for Augmented Dataset Model", fontsize=14, pad=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix saved to {save_path}")


def plot_roc_curves(y_true_onehot: np.ndarray, y_probs: np.ndarray, save_path: Path):
    """Plots multi-class ROC curves and AUC for each lesion category (Paper Fig. 10)."""
    short_names = list(CLASS_MAPPING.keys())
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]

    plt.figure(figsize=(9, 7))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"ROC curve of class {short_names[i]} (area = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Multi-class ROC and AUC", fontsize=14, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Multi-class ROC curves saved to {save_path}")


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device = DEVICE,
    output_dir: Path = OUTPUTS_DIR,
):
    """
    Evaluates test set performance across all paper metrics:
    - Accuracy, Precision, Recall, F1
    - Confusion Matrix
    - Multiclass ROC-AUC
    - MSE, RMSE, MAE on probability distribution
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, y_probs = run_inference(model, test_loader, device)

    # 1. Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    short_names = list(CLASS_MAPPING.keys())
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=short_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=short_names,
        zero_division=0,
    )

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, short_names, output_dir / "confusion_matrix.png")

    # 3. One-hot true labels for ROC and Regression error
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]

    # 4. ROC-AUC Curves
    plot_roc_curves(y_true_onehot, y_probs, output_dir / "roc_curves.png")

    # 5. Regression Metrics (Paper Eq. 24, 25, 26: MAE, MSE, RMSE)
    mse = float(mean_squared_error(y_true_onehot, y_probs))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_onehot, y_probs))

    results = {
        "test_accuracy": float(acc),
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "classification_report": report_dict,
    }

    # Save metrics JSON & Classification Report CSV
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(output_dir / "classification_report.csv")

    print("\n================ Evaluation Results ================")
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("====================================================\n")

    return results


def load_best_model_and_evaluate(
    test_loader,
    checkpoint_path: Path = CHECKPOINTS_DIR / "best_model.pth",
    use_batch_norm: bool = False,
    device: torch.device = DEVICE,
):
    """Helper to load checkpointed weights and run evaluation."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = build_model(use_batch_norm=use_batch_norm).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[*] Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')})")

    return evaluate_model(model, test_loader, device=device)
