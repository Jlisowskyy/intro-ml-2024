"""
Author: Łukasz Kryczka, 2024

This module provides functionality for normalizing audio signals.
"""

import librosa.feature
import numpy as np

from src.constants import (NormalizationType, EPSILON, NORMALIZATION_PCEN_TIME_CONSTANT, \
                           NORMALIZATION_PCEN_ALPHA, NORMALIZATION_PCEN_DELTA, \
                           NORMALIZATION_PCEN_R, NORMALIZATION_PCEN_HOP_LENGTH)


def mean_variance_normalization(signal: np.ndarray) -> np.ndarray:
    """
    Apply mean and variance normalization to the signal.
    Adjusts the signal to have a mean of 0 and a standard deviation of 1.

    :param signal: Input audio signal (numpy array)
    :return: Normalized audio signal
    """

    mean = np.mean(signal)
    std = np.std(signal)
    std = np.maximum(std, EPSILON)
    normalized_signal = (signal - mean) / std
    return normalized_signal

def pcen_normalization(signal: np.ndarray,
                       fs: float,
                       time_constant: float = NORMALIZATION_PCEN_TIME_CONSTANT,
                       alpha: float = NORMALIZATION_PCEN_ALPHA,
                       delta: float = NORMALIZATION_PCEN_DELTA,
                       r: float = NORMALIZATION_PCEN_R,
                       eps: float = EPSILON) -> np.ndarray:
    # pylint: disable=line-too-long
    """
    Apply Per-Channel Energy Normalization (PCEN) to the signal.

    Source: https://bioacoustics.stackexchange.com/questions/846/should-we-normalize-audio-before-training-a-ml-model

    :param signal: Input audio signal (numpy array)
    :param fs: Sampling rate
    :param time_constant: Time constant for the PCEN filter
    :param alpha: Gain factor for the PCEN filter
    :param delta: Bias for the PCEN filter
    :param r: Exponent for the PCEN filter
    :param eps: Small constant to avoid division by zero
    :return: PCEN normalized audio signal (numpy array)
    """

    s = np.abs(librosa.stft(signal)) ** 2
    m = librosa.pcen(s, sr=fs, hop_length=NORMALIZATION_PCEN_HOP_LENGTH,
                     gain=alpha, bias=delta,
                     eps=eps, power=r,
                     time_constant=time_constant)
    normalized_signal = librosa.istft(m)
    return normalized_signal


def normalize(signal: np.ndarray,
              fs: float,
              normalization_type: NormalizationType,
              *args) -> np.ndarray:
    """
    General normalize function that chooses between currently implemented normalization methods.

    :param signal: Input audio signal (numpy array)
    :param fs: Sampling rate
    :param normalization_type: Enum for normalization type (mean-variance, PCEN, or CMVN)
    :param args: Additional arguments to pass to the normalization functions
    :return: Normalized audio signal (numpy array)
    """

    assert signal.dtype in (np.float32, np.float64)

    if normalization_type == NormalizationType.MEAN_VARIANCE:
        return mean_variance_normalization(signal)
    if normalization_type == NormalizationType.PCEN:
        return pcen_normalization(signal, fs, *args)

    raise ValueError("Unsupported normalization type")
