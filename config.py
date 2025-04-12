# config.py
import torch
# Data parameters
DATA_ROOT = '/Users/vighneshnama/Downloads/UCMerced_LandUse'  # Update with the correct path to your dataset
IMAGE_SIZE = 256  # assuming square images; the UC Merced images are 256x256
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training parameters
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TRAIN_VAL_SPLIT = 0.8  # 80% training + validation; later split into train/val
VAL_SPLIT = 0.1        # proportion for validation from whole data

# Model parameters
NUM_CLASSES = 21

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
