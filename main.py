"""
main.py
Main entry point for Skin Lesion Classification.
Supports:
1. Custom CNN (Paper architecture from scratch, 132k params)
2. EfficientNet-B0 (Transfer learning with ImageNet pre-trained weights, ~5.3M params)
3. Model comparison table generator (--compare)

Usage:
    # Run Custom CNN (Baseline from paper)
    python main.py --model custom_cnn --img-size 128 --batch-norm --smooth-balance --epochs 50

    # Run EfficientNet-B0 Transfer Learning
    python main.py --model efficientnet --img-size 224 --epochs 15 --lr 3e-4 --smooth-balance

    # Compare both models side-by-side
    python main.py --compare
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from src.config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    DATA_DIR,
    DEVICE,
)
from src.dataset import get_dataloaders
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, load_best_model_and_evaluate, compare_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification: Custom CNN vs. EfficientNet-B0"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "efficientnet"],
        help="Model architecture: 'custom_cnn' (paper baseline) or 'efficientnet' (transfer learning)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom dataset directory path (auto-detects Kaggle input if not set)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Input image resolution (default: 128 for custom_cnn, 224 for efficientnet)",
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Enable Batch Normalization in custom_cnn (Eq. 10-12 in paper)",
    )
    parser.add_argument(
        "--smooth-balance",
        action="store_true",
        help="Cap minority oversampling at 2500 samples/class to prevent majority false alarms",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Explicit number of samples per class for training balancing",
    )
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Apply class-frequency inverse square root weights to CrossEntropyLoss",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 50 for custom_cnn, 15 for efficientnet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial learning rate (default: 1e-3 for custom_cnn, 3e-4 for efficientnet)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2 if torch.cuda.is_available() else 0,
        help="Number of DataLoader worker processes (default: 2 on GPU, 0 on CPU)",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable training data balancing/augmentation",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and run evaluation on the best saved checkpoint",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate a side-by-side comparison table of all evaluated models",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If --compare is requested, print comparison and exit
    if args.compare:
        compare_models(output_dir=OUTPUTS_DIR)
        return

    model_name = args.model.lower()
    is_efficientnet = "efficientnet" in model_name

    # Set sensible defaults per architecture
    img_size = args.img_size if args.img_size is not None else (224 if is_efficientnet else 128)
    epochs = args.epochs if args.epochs is not None else (15 if is_efficientnet else EPOCHS)
    lr = args.lr if args.lr is not None else (3e-4 if is_efficientnet else LEARNING_RATE)
    normalize_imagenet = is_efficientnet

    data_dir_path = Path(args.data_dir) if args.data_dir else DATA_DIR

    # Determine balancing target
    target_samples = args.target_samples
    if args.smooth_balance and target_samples is None:
        target_samples = 2500

    print("================================================================")
    print("  Skin Cancer Diagnosis - HAM10000 Classification")
    print(f"  Model: {model_name.upper()}")
    print(f"  Device: {DEVICE} (GPUs available: {torch.cuda.device_count()})")
    print(f"  Image Resolution: {img_size}x{img_size}")
    print(f"  Normalization: {'ImageNet (mean/std)' if normalize_imagenet else '[0, 1] scaling'}")
    print(f"  Data Directory: {data_dir_path}")
    print("================================================================")

    # 1. Load DataLoaders
    print("\n[Step 1/3] Preparing datasets and DataLoaders...")
    train_loader, val_loader, test_loader, (train_df, val_df, test_df) = get_dataloaders(
        data_dir=data_dir_path,
        batch_size=args.batch_size,
        balance_train=not args.no_balance,
        target_samples_per_class=target_samples,
        img_size=img_size,
        normalize_imagenet=normalize_imagenet,
        num_workers=args.num_workers,
    )
    print(f"Data ready: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Compute class weights if requested
    class_weights = None
    if args.weighted_loss:
        counts = train_df["label"].value_counts().sort_index().values
        w = 1.0 / np.sqrt(counts)
        w = w / w.sum() * len(counts)
        class_weights = torch.tensor(w, dtype=torch.float32)
        print(f"[*] Computed class weights: {class_weights.tolist()}")

    # 2. Train or Load Checkpoint
    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"

    if not args.evaluate_only:
        print(f"\n[Step 2/3] Building {model_name} and starting training...")
        model = build_model(
            model_name=model_name,
            use_batch_norm=args.batch_norm,
            pretrained=True,
        ).to(DEVICE)
        print(f"Model built: {model.count_parameters():,} trainable parameters.")

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            epochs=epochs,
            lr=lr,
            class_weights=class_weights,
            device=DEVICE,
            checkpoint_dir=CHECKPOINTS_DIR,
            output_dir=OUTPUTS_DIR,
        )
    else:
        print(f"\n[Step 2/3] Skipping training (--evaluate-only specified).")

    # 3. Final Evaluation on Test Set using Best Saved Model
    print(f"\n[Step 3/3] Running final evaluation on Test Set for {model_name}...")
    load_best_model_and_evaluate(
        test_loader=test_loader,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        use_batch_norm=args.batch_norm,
        device=DEVICE,
    )

    # 4. Generate comparison if other model results exist
    compare_models(output_dir=OUTPUTS_DIR)


if __name__ == "__main__":
    main()
