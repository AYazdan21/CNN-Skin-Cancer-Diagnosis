"""
src/dataset.py
Data preprocessing, class balancing, augmentation, and PyTorch Dataset / DataLoader creation.
Compatible with local and Kaggle environments.
"""

from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    DATA_DIR,
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


def find_metadata_file(data_dir: Path) -> Path:
    """Finds the metadata file whether named HAM10000_metadata or HAM10000_metadata.csv."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    # 1. Direct candidate checks
    candidates = [
        data_dir / "HAM10000_metadata",
        data_dir / "HAM10000_metadata.csv",
        data_dir / "ham10000_metadata.csv",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # 2. Search recursively for any file containing 'metadata' or any '.csv' with 'dx' column
    found_csvs = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*metadata*"))
    for file_path in found_csvs:
        if file_path.is_file():
            try:
                head = pd.read_csv(file_path, nrows=2)
                cols = [str(col).strip().lower() for col in head.columns]
                if "dx" in cols and "image_id" in cols:
                    return file_path
            except Exception:
                continue

    # 3. Fallback: Search across all of /kaggle/input (ignoring 'notebooks')
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists() and data_dir != kaggle_input:
        for file_path in list(kaggle_input.rglob("*.csv")) + list(kaggle_input.rglob("*metadata*")):
            if "notebooks" in str(file_path):
                continue
            if file_path.is_file():
                try:
                    head = pd.read_csv(file_path, nrows=2)
                    cols = [str(col).strip().lower() for col in head.columns]
                    if "dx" in cols and "image_id" in cols:
                        print(f"[*] Auto-discovered metadata in Kaggle dataset: {file_path}")
                        return file_path
                except Exception:
                    continue

    # Diagnostic info if not found
    items = list(data_dir.iterdir()) if data_dir.is_dir() else []
    item_names = [i.name for i in items]
    kaggle_items = [i.name for i in kaggle_input.iterdir()] if kaggle_input.exists() else []

    raise FileNotFoundError(
        f"Could not find HAM10000 metadata CSV in '{data_dir}'.\n"
        f"Folders in '/kaggle/input': {kaggle_items}\n"
        f"NOTE: You added a Notebook ('kmader') rather than the HAM10000 Dataset!\n"
        f"-> In Kaggle, click '+ Add Input' -> search for 'Skin Cancer MNIST: HAM10000' -> click Add."
    )


def find_image_files(data_dir: Path) -> dict:
    """Finds all images in data_dir and maps image stem (e.g. ISIC_0027419) to file path."""
    image_paths = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for p in Path(data_dir).rglob("*"):
        if "notebooks" in str(p):
            continue
        if p.suffix in valid_exts and p.is_file():
            image_paths[p.stem] = str(p)

    # Fallback to search /kaggle/input if data_dir had no images
    kaggle_input = Path("/kaggle/input")
    if len(image_paths) == 0 and kaggle_input.exists() and data_dir != kaggle_input:
        for p in kaggle_input.rglob("*"):
            if "notebooks" in str(p):
                continue
            if p.suffix in valid_exts and p.is_file():
                image_paths[p.stem] = str(p)

    return image_paths


def load_and_prepare_metadata(data_dir: Path = None) -> pd.DataFrame:
    """
    Loads metadata, imputes missing values (e.g. age), maps image IDs to file paths,
    and maps diagnoses to numeric class labels.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)
    metadata_path = find_metadata_file(data_dir)
    print(f"[*] Found metadata file: {metadata_path}")

    # 1. Read metadata
    df = pd.read_csv(metadata_path)

    # 2. Impute missing values (Paper Section: Preprocessing / Algorithm 1)
    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].mean())

    # 3. Map image_id to actual image file paths on disk
    image_paths = find_image_files(data_dir)
    print(f"[*] Found {len(image_paths)} total images in {data_dir}")

    if len(image_paths) == 0:
        raise FileNotFoundError(f"Found metadata at {metadata_path}, but 0 image files were found in {data_dir}!")

    df["image_path"] = df["image_id"].map(image_paths)
    missing_paths = df["image_path"].isnull().sum()
    if missing_paths > 0:
        # If some images aren't present (e.g. subset), drop rows that don't have images
        if missing_paths == len(df):
            raise FileNotFoundError(
                f"None of the {len(df)} image IDs matched the image files found in {data_dir}."
            )
        else:
            print(f"[!] Warning: {missing_paths} out of {len(df)} images missing. Filtering to {len(df) - missing_paths} available images.")
            df = df.dropna(subset=["image_path"]).reset_index(drop=True)

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


def get_transforms(img_size: int = IMAGE_HEIGHT, normalize_imagenet: bool = False):
    """
    Returns data transforms for training and validation/testing.
    Training uses rotation, zoom, flips, shearing, and brightness adjustment.
    If normalize_imagenet=True, applies ImageNet mean & std (recommended for EfficientNet).
    Otherwise scales pixels to [0, 1] via ToTensor() (paper Eq. 2).
    """
    train_ops = [
        transforms.Resize((img_size, img_size)),
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
        transforms.ToTensor(),
    ]

    eval_ops = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    if normalize_imagenet:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_ops.append(norm)
        eval_ops.append(norm)

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def get_dataloaders(
    data_dir: Path = None,
    batch_size: int = BATCH_SIZE,
    balance_train: bool = True,
    target_samples_per_class: int = None,
    img_size: int = IMAGE_HEIGHT,
    normalize_imagenet: bool = False,
    num_workers: int = 0,
):
    """
    High-level factory function:
    1. Loads metadata and resolves paths
    2. Splits into stratified train, val, and test sets
    3. Balances the train split using oversampling
    4. Builds PyTorch DataLoaders with given img_size and normalization
    """
    df = load_and_prepare_metadata(data_dir=data_dir)
    train_df, val_df, test_df = split_data(df)

    if balance_train:
        train_df = balance_training_data(
            train_df,
            target_samples_per_class=target_samples_per_class,
        )

    train_transform, eval_transform = get_transforms(
        img_size=img_size,
        normalize_imagenet=normalize_imagenet,
    )

    train_dataset = HAM10000Dataset(train_df, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, transform=eval_transform)
    test_dataset = HAM10000Dataset(test_df, transform=eval_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, (train_df, val_df, test_df)
