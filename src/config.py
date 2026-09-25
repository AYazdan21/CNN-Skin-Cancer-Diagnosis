"""
src/config.py
Configuration file containing paths, hyperparameters, and class mappings based on:
'Enhanced skin cancer diagnosis using optimized CNN architecture and checkpoints for automated dermatological lesion classification'
"""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataverse_files"

# Dataset paths
METADATA_PATH = DATA_DIR / "HAM10000_metadata"
IMAGES_DIR_PART1 = DATA_DIR / "HAM10000_images_part_1"
IMAGES_DIR_PART2 = DATA_DIR / "HAM10000_images_part_2"

# Output directories
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Image settings (from paper Section 'Image resizing' & Table 2)
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 3

# Data split ratios (Paper: 80% train, 10% validation, 10% test)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

# Training Hyperparameters (Paper: Adam, batch size 128, 50 epochs)
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5  # L2 regularization mentioned in paper
DROPOUT_RATE = 0.3   # Dropout layer between dense layers

# Callbacks settings (Paper: ReduceLROnPlateau and EarlyStopping)
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# 7 Lesion Classes in HAM10000
CLASS_MAPPING = {
    'nv': 0,     # Melanocytic nevi
    'mel': 1,    # Melanoma
    'bkl': 2,    # Benign keratosis-like lesions
    'bcc': 3,    # Basal cell carcinoma
    'akiec': 4,  # Actinic keratoses and intraepithelial carcinoma
    'vasc': 5,   # Vascular lesions
    'df': 6      # Dermatofibroma
}

CLASS_NAMES = [
    'Melanocytic nevi (nv)',
    'Melanoma (mel)',
    'Benign keratosis (bkl)',
    'Basal cell carcinoma (bcc)',
    'Actinic keratoses (akiec)',
    'Vascular lesions (vasc)',
    'Dermatofibroma (df)'
]

NUM_CLASSES = len(CLASS_MAPPING)

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
