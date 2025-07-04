"""
Global configuration for the project.
"""

import os
from pathlib import Path

# Data paths
DATA_ROOT = Path("./data")
TRAIN_IMG_DIR = DATA_ROOT / "X_train"
TRAIN_MASK_DIR = DATA_ROOT / "y_train"
VAL_IMG_DIR = DATA_ROOT / "X_val"
VAL_MASK_DIR = DATA_ROOT / "y_val"
TEST_IMG_DIR = DATA_ROOT / "X_test"
TEST_MASK_DIR = DATA_ROOT / "y_test"

# Model parameters
NUM_CLASSES = 12
PATCH_SIZE = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda"  # or "cpu"

# TRM parameters
HIDDEN_DIM = 128
GNN_LAYERS = 3
ENABLE_GLOBAL_TISSUE_EMBEDDINGS = True

# Color mapping for visualization
COLOR_DICT = {
    "EPI":  [73, 0, 106],
    "GLD":  [108, 0, 115],
    "INF":  [145, 1, 122],
    "RET":  [181, 9, 130],
    "FOL":  [216, 47, 148],
    "PAP":  [236, 85, 157],
    "HYP":  [254, 246, 242],
    "KER":  [248, 123, 168],
    "BKG":  [0, 0, 0],
    "BCC":  [127, 255, 255],
    "SCC":  [127, 255, 142],
    "IEC":  [255, 127, 127]
}

CLASS_NAMES = list(COLOR_DICT.keys())
CLASS_COLORS = list(COLOR_DICT.values())

# Paths for saving
CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./logs")
RESULTS_DIR = Path("./results")

# Create directories if they don't exist
for dir_path in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)
    