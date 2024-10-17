"""
Author: Łukasz Kryczka, 2024

This module provides functionality for normalizing audio signals.
"""

from enum import Enum

import librosa.feature
import numpy as np


class NormalizationType(Enum):
    """
    Enum for different types of normalization.
    """

    MEAN_VARIANCE = 1
    PCEN = 2
    CMVN = 3


def mean_variance_normalization(signal: np.ndarray) -> np.ndarray:
    """
    Apply mean and variance normalization to the signal.
    Adjusts the signal to have a mean of 0 and a standard deviation of 1.

    :param signal: Input audio signal (numpy array)
    :return: Normalized audio signal
    """

    mean = np.mean(signal)
    std = np.std(signal)
    std = np.maximum(std, 1e-8)
    normalized_signal = (signal - mean) / std
    return normalized_signal


def cmvn_normalization(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply Cepstral Mean and Variance Normalization (CMVN) to the signal.

    Source: https://en.wikipedia.org/wiki/Cepstral_mean_and_variance_normalization

    :param signal: Input audio signal (numpy array)
    :param fs: Sampling rate
    :return: CMVN normalized audio signal (numpy array)
    """

    mfccs = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13)

    mfccs_mean = np.mean(mfccs, axis=1, keepdims=True)
    mfccs_std = np.std(mfccs, axis=1, keepdims=True)

    mfccs_normalized = (mfccs - mfccs_mean) / (mfccs_std + 1e-8)

    normalized_signal = librosa.feature.inverse.mfcc_to_audio(mfccs_normalized, sr=fs)

    return normalized_signal


def pcen_normalization(signal: np.ndarray,
                       fs: float,
                       time_constant: float = 0.06,
                       alpha: float = 0.98,
                       delta: float = 2,
                       r: float = 0.5,
                       eps: float = 1e-6) -> np.ndarray:
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
    m = librosa.pcen(s, sr=fs, hop_length=512,
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

    if normalization_type == NormalizationType.MEAN_VARIANCE:
        return mean_variance_normalization(signal)
    if normalization_type == NormalizationType.PCEN:
        return pcen_normalization(signal, fs, *args)
    if normalization_type == NormalizationType.CMVN:
        return cmvn_normalization(signal, fs)

    raise ValueError("Unsupported normalization type")
