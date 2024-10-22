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

TRAINING_TEST_BATCH_SIZE: int = 128
TRAINING_VALIDATION_BATCH_SIZE: int = 128
TRAINING_EPOCHS: int = 10
TRAINING_LEARNING_RATES: list[float] = [0.001]
TRAINING_TRAIN_SET_SIZE: float = 0.8
TRAINING_TEST_SET_SIZE: float = 0.2

assert abs(TRAINING_TRAIN_SET_SIZE + TRAINING_TEST_SET_SIZE - 1) < 1e-6, \
    "Train and test set sizes should sum to 1"

TRAINING_MOMENTUM: float = 0.9

# ------------------------------
# SPECTROGRAM constants
# ------------------------------

SPECTROGRAM_WIDTH: int = 400
SPECTROGRAM_HEIGHT: int = 300

# ------------------------------
# DATABASE constants
# ------------------------------

DATABASE_CUT_ITERATOR: WavIteratorType = WavIteratorType.PLAIN
DATABASE_PATH: str = './datasets/daps'
DATABASE_OUT_NAME: str = 'daps_split_spectro'

# ------------------------------
# MODEL constants
# ------------------------------

MODEL_WINDOW_LENGTH: int = 5
