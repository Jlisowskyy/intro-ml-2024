"""
Author: Jakub Lisowski, 2024

File collects various constants used in the project as well as enums used for configuration.
"""

from enum import Enum, IntEnum


# ------------------------------
# TYPE ENUMS
# ------------------------------

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
# GENERAL constants
# ------------------------------

EPSILON: float = 1e-8
"""
Small constant to avoid division by zero
"""

# ------------------------------
# DENOISE constants
# ------------------------------

DENOISE_FREQ_HIGH_CUT: float = 8200.0
DENOISE_FREQ_LOW_CUT: float = 80.0
"""
Highcut is chosen to be 8200 Hz : common male and female voices frequency range
"""

# ------------------------------
# DETECT SPEECH constants
# ------------------------------

DETECT_SILENCE_WINDOW_MAX_MS : int = 50
DETECT_SILENCE_THRESHOLD_DB: int = -40
SILENCE_CUT_WINDOW_MS: int = 25

# ------------------------------
# NORMALIZE constants
# ------------------------------


NORMALIZATION_TYPE: NormalizationType = NormalizationType.MEAN_VARIANCE

NORMALIZATION_PCEN_TIME_CONSTANT: float = 0.06
"""
Time constant for the PCEN filter, controls the smoothing of the signal.
"""

NORMALIZATION_PCEN_ALPHA: float = 0.98
"""
Gain factor for the PCEN filter, determines the strength of the gain normalization.
"""

NORMALIZATION_PCEN_DELTA: float = 2.0
"""
Bias for the PCEN filter, added to the signal to avoid taking the logarithm of zero.
"""

NORMALIZATION_PCEN_R: float = 0.5
"""
Exponent for the PCEN filter, controls the compression of the signal.
"""
NORMALIZATION_PCEN_HOP_LENGTH: int = 512

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
SPECTROGRAM_DPI: int = 100
SPECTROGRAM_N_FFT: int = 4096
SPECTROGRAM_HOP_LENGTH: int = 512
SPECTROGRAM_N_MELS: int = 256

# ------------------------------
# WAV ITERATOR constants
# ------------------------------
WINDOW_SIZE_FRAMES_DIVISOR: int = 10
"""
Divisor to calculate the window size in frames for WAV iterators.
"""

# ------------------------------
# DATABASE constants
# ------------------------------

SPEAKER_CLASSES = {
    'm1': 0,
    'm2': 0,
    'm3': 1,
    'm4': 0,
    'm5': 0,
    'm6': 1,
    'm7': 0,
    'm8': 1,
    'm9': 0,
    'm10': 0,
    'f1': 1,
    'f2': 0,
    'f3': 0,
    'f4': 0,
    'f5': 0,
    'f6': 0,
    'f7': 1,
    'f8': 1,
    'f9': 0,
    'f10': 0
}

CLASSES = [
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go',
    'unknown',
    'silence'
]

DATABASE_CUT_ITERATOR: WavIteratorType = WavIteratorType.PLAIN
DATABASE_NAME: str = 'kaggle'
DATABASE_OUT_NAME: str = 'kaggle_spectro'
DATABASE_PATH: str = f'./datasets/{DATABASE_NAME}'
DATABASE_OUT_PATH: str = f'./datasets/{DATABASE_OUT_NAME}'
DATABASE_ANNOTATIONS_PATH: str = './annotations_kaggle.csv'
DATABASE_VALID_WAV_SR: int = 16000

# ------------------------------
# MODEL constants
# ------------------------------

MODEL_WINDOW_LENGTH: int = 1
MODEL_BASE_PATH: str = './models/model.pth'
MODEL_PRETRAINED_PATH: str = './models/pretrained.pth'

CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.7
"""
How many chunks need to be classified as 1 to classify the whole file as 1
"""

# ------------------------------
# HELPER SCRIPTS constants
# ------------------------------

# From Wav to Histogram
HELPER_SCRIPTS_SPECTROGRAM_FOLDER_SUFFIX: str = '_spectrograms'
HELPER_SCRIPTS_HISTOGRAM_DEFAULT_DIR: str = 'work_dir'
HELPER_SCRIPTS_HISTOGRAM_N_BINS: int = 256
HELPER_SCRIPTS_HISTOGRAM_ALPHA: float = 0.5
