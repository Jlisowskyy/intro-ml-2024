"""
Author: Jakub Lisowski, 2024

File collects various constants used in the project as well as enums used for configuration.
"""

from enum import Enum, IntEnum

# ------------------------------
# TYPE ENUMS
# ------------------------------

class DenoiseType(Enum):
    """
    Enum for different types of denoising.
    Future types of denoising can be added here and
    handled in the denoise function
    """

    BASIC = 1


class NormalizationType(Enum):
    """
    Enum for different types of normalization.
    Future types of normalization can be added here and
    handled in the normalize function
    """

    MEAN_VARIANCE = 1
    PCEN = 2
    CMVN = 3


class WavIteratorType(IntEnum):
    """
    Enumeration of the available WAV file iterators
    """

    PLAIN = 0
    OVERLAPPING = 1

# ------------------------------
# DENOISE constants
# ------------------------------

DENOISE_TYPE: DenoiseType = DenoiseType.BASIC

DENOISE_FREQ_LOW_CUT: float = 50.0
"""
Lowcut is chosen to be 50 Hz : Male voice frequency range
"""

DENOISE_FREQ_HIGH_CUT: float = 8200.0
"""
Highcut is chosen to be 8200 Hz : common male and female voices frequency range
"""

# ------------------------------
# DETECT SPEECH constants
# ------------------------------

DETECT_SILENCE_TOLERANCE: float = 0.5
DETECT_SILENCE_THRESHOLD: float = 0.015

# ------------------------------
# NORMALIZE constants
# ------------------------------

NORMALIZATION_TYPE: NormalizationType = NormalizationType.MEAN_VARIANCE

# ------------------------------
# TRAINING constants
# ------------------------------

TRAINING_TRAIN_BATCH_SIZE: int = 128
TRAINING_VALIDATION_BATCH_SIZE: int = 128
TRAINING_TEST_BATCH_SIZE: int = 128
TRAINING_EPOCHS: int = 10
# 0.1 seems to be too high (exploding loss)
# this porridge is pretty decent (maybe should be smaller? TODO: check)
TRAINING_LEARNING_RATES: list[float] = [0.0001]
TRAINING_TRAIN_SET_SIZE: float = 0.64
TRAINING_VALIDATION_SET_SIZE: float = 0.16
TRAINING_TEST_SET_SIZE: float = 0.2

# torch split does *not* like epsilon, requires the sum to be exactly 1.0
assert (TRAINING_TRAIN_SET_SIZE +
           TRAINING_VALIDATION_SET_SIZE +
           TRAINING_TEST_SET_SIZE == 1.0), \
    "All set sizes should sum to 1"

TRAINING_MOMENTUM: float = 0.9

# ------------------------------
# SPECTROGRAM constants
# ------------------------------

SPECTROGRAM_WIDTH: int = 300
SPECTROGRAM_HEIGHT: int = 400

# ------------------------------
# DATABASE constants
# ------------------------------

CLASSES = {
    'm1': 0,
    'm2': 0,
    'm3': 1,
    'm4': 0,
    'm5': 0,
    'm6': 1,
    'm7': 0,
    'm8': 1,
    'm9': 0,
    'm10':0,
    'f1': 1,
    'f2': 0,
    'f3': 0,
    'f4': 0,
    'f5': 0,
    'f6': 0,
    'f7': 1,
    'f8': 1,
    'f9': 0,
    'f10':0
}

DATABASE_CUT_ITERATOR: WavIteratorType = WavIteratorType.PLAIN
DATABASE_PATH: str = './datasets/daps'
DATABASE_OUT_NAME: str = 'daps_split_spectro'
DATABASE_OUT_PATH: str = f'./datasets/{DATABASE_OUT_NAME}'
DATABASE_ANNOTATIONS_PATH: str = './annotations.csv'

SPEAKER_TO_CLASS = {
    'f1': 1,
    'f7': 1,
    'f8': 1,
    'm3': 1,
    'm6': 1,
    'm8': 1
}

# ------------------------------
# MODEL constants
# ------------------------------

MODEL_WINDOW_LENGTH: int = 3
MODEL_BASE_PATH: str = './models/model.pth'
