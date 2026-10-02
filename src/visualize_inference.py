"""
src/visualize_inference.py
Generates publication-quality sample prediction grids for README and documentation:
1. Correct predictions grid (reproducing Paper Fig. 7: 'Predicted label with images')
2. Misclassified / ambiguous cases grid (reproducing Paper Fig. 12: 'Misclassified instances')
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import CLASS_MAPPING, OUTPUTS_DIR, DEVICE


def denormalize_image(tensor: torch.Tensor, is_imagenet: bool = False) -> np.ndarray:
    """Converts a PyTorch tensor (C, H, W) to a displayable NumPy image (H, W, C) in [0, 1]."""
    img = tensor.clone().detach().cpu()
    if is_imagenet:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean

    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def generate_inference_visualizations(
    model: torch.nn.Module,
    test_loader,
    model_name: str = "custom_cnn",
    is_imagenet: bool = False,
    device: torch.device = DEVICE,
    output_dir: Path = OUTPUTS_DIR,
    num_correct: int = 8,
    num_errors: int = 4,
):
    """
    Collects test predictions and renders:
    - sample_predictions_{model}.png (Fig. 7 in paper)
    - misclassified_samples_{model}.png (Fig. 12 in paper)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    short_names = list(CLASS_MAPPING.keys())

    correct_samples = []
    error_samples = []

    with torch.no_grad():
        for images, targets in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)

            for i in range(images.size(0)):
                img_cpu = images[i]
                true_cls = targets[i].item()
                pred_cls = preds[i].item()
                conf = confidences[i].item()

                item = {
                    "image": img_cpu,
                    "true_label": short_names[true_cls],
                    "pred_label": short_names[pred_cls],
                    "confidence": conf,
                }

                if true_cls == pred_cls:
                    if len(correct_samples) < num_correct:
                        correct_samples.append(item)
                else:
                    if len(error_samples) < num_errors:
                        error_samples.append(item)

                if len(correct_samples) >= num_correct and len(error_samples) >= num_errors:
                    break

            if len(correct_samples) >= num_correct and len(error_samples) >= num_errors:
                break

    # 1. Plot Correct Predictions Grid (reproducing Paper Fig. 7)
    if correct_samples:
        ncols = 4
        nrows = int(np.ceil(len(correct_samples) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
        axes = np.array(axes).reshape(-1)

        for idx, item in enumerate(correct_samples):
            ax = axes[idx]
            display_img = denormalize_image(item["image"], is_imagenet=is_imagenet)
            ax.imshow(display_img)
            ax.set_title(
                f"True: {item['true_label']} | Pred: {item['pred_label']}\nConf: {item['confidence']*100:.1f}%",
                fontsize=11,
                color="#006400",
                fontweight="bold",
                pad=6,
            )
            ax.axis("off")

        # Hide any unused subplots
        for idx in range(len(correct_samples), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Sample Model Predictions ({model_name}) - Fig. 7 Reproduction",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        save_path = output_dir / f"sample_predictions_{model_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[*] Saved sample predictions figure to {save_path}")

    # 2. Plot Misclassified Predictions Grid (reproducing Paper Fig. 12)
    if error_samples:
        ncols = len(error_samples)
        fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.6))
        if ncols == 1:
            axes = [axes]

        for idx, item in enumerate(error_samples):
            ax = axes[idx]
            display_img = denormalize_image(item["image"], is_imagenet=is_imagenet)
            ax.imshow(display_img)
            ax.set_title(
                f"True: {item['true_label']}\nPred: {item['pred_label']} ({item['confidence']*100:.1f}%)",
                fontsize=11,
                color="#B22222",
                fontweight="bold",
                pad=6,
            )
            ax.axis("off")

        plt.suptitle(
            f"Misclassified Cases ({model_name}) - Fig. 12 Reproduction",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        save_path = output_dir / f"misclassified_samples_{model_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[*] Saved misclassified samples figure to {save_path}")
