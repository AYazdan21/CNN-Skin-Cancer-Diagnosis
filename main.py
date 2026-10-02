"""
main.py
Main entry point for Skin Lesion Classification.
Allows running training, evaluation, or both from the command line.
Supports local execution and Kaggle Multi-GPU environments.

Usage:
    # Baseline run
    python main.py --epochs 50 --batch-size 128

    # Higher resolution + BatchNorm + smooth balancing (recommended for pushing accuracy to 76%-80%)
    python main.py --img-size 64 --batch-norm --smooth-balance --epochs 50

    # Evaluate existing checkpoint
    python main.py --evaluate-only
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
    NUM_CLASSES,
)
from src.dataset import get_dataloaders
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, load_best_model_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN Skin Cancer Diagnosis based on BMC Medical Imaging (2024)"
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
        default=28,
        help="Input image resolution: 28 (paper baseline), 64, or 128 (default: 28)",
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Enable Batch Normalization layers in CNN (paper Equations 10-12)",
    )
    parser.add_argument(
        "--smooth-balance",
        action="store_true",
        help="Use moderate/smooth class oversampling (cap at 2500 samples/class) to avoid over-predicting false positives on majority class",
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
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
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
        default=LEARNING_RATE,
        help=f"Initial learning rate (default: {LEARNING_RATE})",
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
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir_path = Path(args.data_dir) if args.data_dir else DATA_DIR

    # Determine balancing target
    target_samples = args.target_samples
    if args.smooth_balance and target_samples is None:
        target_samples = 2500

    print("================================================================")
    print("  CNN Skin Cancer Diagnosis - HAM10000 Classification")
    print(f"  Device: {DEVICE} (GPUs available: {torch.cuda.device_count()})")
    print(f"  Image Resolution: {args.img_size}x{args.img_size}")
    print(f"  Batch Normalization: {args.batch_norm}")
    print(f"  Data Directory: {data_dir_path}")
    print("================================================================")

    # 1. Load DataLoaders
    print("\n[Step 1/3] Preparing datasets and DataLoaders...")
    train_loader, val_loader, test_loader, (train_df, val_df, test_df) = get_dataloaders(
        data_dir=data_dir_path,
        batch_size=args.batch_size,
        balance_train=not args.no_balance,
        target_samples_per_class=target_samples,
        img_size=args.img_size,
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
    checkpoint_path = CHECKPOINTS_DIR / "best_model.pth"
    if not args.evaluate_only:
        print("\n[Step 2/3] Building model and starting training...")
        model = build_model(use_batch_norm=args.batch_norm).to(DEVICE)
        print(f"Model built: {model.count_parameters():,} trainable parameters.")

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            class_weights=class_weights,
            device=DEVICE,
            checkpoint_dir=CHECKPOINTS_DIR,
            output_dir=OUTPUTS_DIR,
        )
    else:
        print(f"\n[Step 2/3] Skipping training (--evaluate-only specified).")

    # 3. Final Evaluation on Test Set using Best Saved Model
    print("\n[Step 3/3] Running final evaluation on Test Set...")
    load_best_model_and_evaluate(
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        use_batch_norm=args.batch_norm,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
