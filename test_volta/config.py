"""File containing configuration parameters."""

from pathlib import Path

# Different types of signal included in the ECG
SIGNAL_TYPES = [
    "avf",
    "avl",
    "avr",
    "i",
    "ii",
    "iii",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
]
# Path to the result folder
RESULT_FOLDER_PATH = Path("./results")
# Sampling frequency of the ECG
SAMPLING_FREQUENCY = 500  # [Hz]
# Which signal to use to cut heartbeats annotations from
SIGNAL_USED_FOR_ANNOTATION = "avf"
# Size of the sliding window used to compute the features in seconds
SLIDING_WINDOW_SIZE = 0.05  # [s]
# Type of segmentation for dataset
#   Either "beat-based" or "patient-based"
SEGMENTATION_TYPE = "patient-based"
# Proportion of the set to use for model training
TRAIN_SIZE = 0.8
# Model to use for computation
MODEL_TYPE = "random_forest"  # 2 possibilities: "random_forest" or "xgboost"
