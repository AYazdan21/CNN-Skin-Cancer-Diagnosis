"""
src/dataset.py
Data preprocessing, class balancing, augmentation, and PyTorch Dataset / DataLoader creation.
Follows the paper:
- Downscales images to 28x28 RGB
- Normalizes pixel values to [0, 1]
- Imputes missing metadata
- Performs stratified 80/10/10 train/val/test split
- Augments training set to balance classes (~6,000 samples per class)
"""

from pathlib import Path
import glob
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    METADATA_PATH,
    IMAGES_DIR_PART1,
    IMAGES_DIR_PART2,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    BATCH_SIZE,
    CLASS_MAPPING,
    NUM_CLASSES,
)


def load_and_prepare_metadata(
    metadata_path: Path = METADATA_PATH,
    part1_dir: Path = IMAGES_DIR_PART1,
    part2_dir: Path = IMAGES_DIR_PART2,
) -> pd.DataFrame:
    """
    Loads metadata, imputes missing values (e.g. age), maps image IDs to file paths,
    and maps diagnoses to numeric class labels.
    """
    # 1. Read metadata
    df = pd.read_csv(metadata_path)

    # 2. Impute missing values (Paper Section: Preprocessing / Algorithm 1)
    df["age"] = df["age"].fillna(df["age"].mean())

    # 3. Map image_id to actual image file paths on disk
    image_paths = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in glob.glob(str(part1_dir / ext)):
            stem = Path(p).stem
            image_paths[stem] = p
        for p in glob.glob(str(part2_dir / ext)):
            stem = Path(p).stem
            image_paths[stem] = p

    df["image_path"] = df["image_id"].map(image_paths)
    missing_paths = df["image_path"].isnull().sum()
    if missing_paths > 0:
        raise FileNotFoundError(
            f"Could not find image files for {missing_paths} metadata entries."
        )

    # 4. Map diagnoses strings to integer labels (0 to 6)
    df["label"] = df["dx"].map(CLASS_MAPPING)

    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
):
    """
    Performs stratified 80% train, 10% validation, 10% test split.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"],
        random_state=random_seed,
    )

    # Second split: val vs test (split val_test 50/50 when val_ratio == test_ratio)
    test_share = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_share,
        stratify=val_test_df["label"],
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def balance_training_data(
    train_df: pd.DataFrame,
    target_samples_per_class: int = None,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Balances the training set across classes by oversampling minority classes
    to reach target_samples_per_class (default: max class count in train split, ~5,364).
    Combined with on-the-fly random transformations in HAM10000Dataset,
    this yields diverse augmented images for all minority classes.
    """
    class_counts = train_df["label"].value_counts()
    if target_samples_per_class is None:
        target_samples_per_class = class_counts.max()

    balanced_dfs = []
    for label in range(NUM_CLASSES):
        class_subset = train_df[train_df["label"] == label]
        n_samples = len(class_subset)
        if n_samples < target_samples_per_class:
            resampled = class_subset.sample(
                n=target_samples_per_class,
                replace=True,
                random_state=random_seed,
            )
            balanced_dfs.append(resampled)
        else:
            balanced_dfs.append(class_subset)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    # Shuffle the balanced dataframe
    balanced_df = balanced_df.sample(
        frac=1.0, random_state=random_seed
    ).reset_index(drop=True)
    return balanced_df


class HAM10000Dataset(Dataset):
    """
    Custom PyTorch Dataset for HAM10000 skin lesion images.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform
        self.image_paths = df["image_path"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_transforms():
    """
    Returns data transforms for training and validation/testing.
    Training uses rotation, zoom, flips, shearing, and brightness adjustment (Paper Sec. Data Augmentation).
    Both resize to 28x28 and scale pixels to [0, 1] via ToTensor().
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.95, 1.05),
            shear=5,
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),  # Scales [0, 255] -> [0.0, 1.0] (Eq. 2 in paper)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),  # Scales [0, 255] -> [0.0, 1.0]
    ])

    return train_transform, eval_transform


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    balance_train: bool = True,
    num_workers: int = 0,
):
    """
    High-level factory function:
    1. Loads metadata and resolves paths
    2. Splits into stratified train, val, and test sets
    3. Balances the train split using oversampling
    4. Builds PyTorch DataLoaders
    """
    df = load_and_prepare_metadata()
    train_df, val_df, test_df = split_data(df)

    if balance_train:
        train_df = balance_training_data(train_df)

    train_transform, eval_transform = get_transforms()

    train_dataset = HAM10000Dataset(train_df, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, transform=eval_transform)
    test_dataset = HAM10000Dataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, (train_df, val_df, test_df)
