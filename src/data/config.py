"""
Central configuration for the Facial Expression Recognition data pipeline.
"""
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "CNN_DataSet")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Dataset Configuration
CLASS_MAPPING = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
CLASSES = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
NUM_CLASSES = len(CLASSES)

# Split Configuration
VALIDATION_RATIO = 0.1
RANDOM_SEED = 42

# Image Configuration
IMAGE_SIZE = (48, 48)
IMAGE_MODE = "L"
CHANNELS = 1
NORMALIZATION_RULE = "pixel_value / 255.0"
