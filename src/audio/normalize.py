"""
Author: Łukasz Kryczka, 2024

This module provides functionality for normalizing audio signals.
"""

import librosa.feature
import numpy as np

from src.audio.audio_data import AudioData
from src.constants import (NormalizationType, EPSILON, NORMALIZATION_PCEN_TIME_CONSTANT,
                           NORMALIZATION_PCEN_ALPHA, NORMALIZATION_PCEN_DELTA,
                           NORMALIZATION_PCEN_R, NORMALIZATION_PCEN_HOP_LENGTH)


def mean_variance_normalization(audio_data: AudioData) -> AudioData:
    """
    Apply mean and variance normalization to the signal.
    Adjusts the signal to have a mean of 0 and a standard deviation of 1.

    :param audio_data: Audio data to be normalized
    :return: Audio data with normalized signal
    """

    mean = np.mean(audio_data.audio_signal)
    std = np.std(audio_data.audio_signal)
    std = np.maximum(std, EPSILON)
    normalized_signal = (audio_data.audio_signal - mean) / std
    audio_data.audio_signal = normalized_signal
    return audio_data


def pcen_normalization(audio_data: AudioData,
                       time_constant: float = NORMALIZATION_PCEN_TIME_CONSTANT,
                       alpha: float = NORMALIZATION_PCEN_ALPHA,
                       delta: float = NORMALIZATION_PCEN_DELTA,
                       r: float = NORMALIZATION_PCEN_R,
                       eps: float = EPSILON) -> AudioData:
    # pylint: disable=line-too-long
    """
    Apply Per-Channel Energy Normalization (PCEN) to the signal.

    Source: https://bioacoustics.stackexchange.com/questions/846/should-we-normalize-audio-before-training-a-ml-model

    :param audio_data: Audio data to be normalized
    :param time_constant: Time constant for the PCEN filter
    :param alpha: Gain factor for the PCEN filter
    :param delta: Bias for the PCEN filter
    :param r: Exponent for the PCEN filter
    :param eps: Small constant to avoid division by zero
    :return: PCEN normalized audio signal (numpy array)
    """

    s = np.abs(librosa.stft(audio_data.audio_signal)) ** 2
    m = librosa.pcen(s, sr=audio_data.sample_rate, hop_length=NORMALIZATION_PCEN_HOP_LENGTH,
                     gain=alpha, bias=delta,
                     eps=eps, power=r,
                     time_constant=time_constant)
    normalized_signal = librosa.istft(m)
    audio_data.audio_signal = normalized_signal
    return audio_data


def normalize(audio_data: AudioData,
              normalization_type: NormalizationType,
              *args) -> AudioData:
    """
    General normalize function that chooses between currently implemented normalization methods.

    :param audio_data: Audio data to be normalized
    :param normalization_type: Enum for normalization type (mean-variance, PCEN, or CMVN)
    :param args: Additional arguments to pass to the normalization functions
    :return: Normalized audio signal (numpy array)
    """

    assert audio_data.audio_signal.dtype in (np.float32, np.float64)

    if normalization_type == NormalizationType.MEAN_VARIANCE:
        return mean_variance_normalization(audio_data)
    if normalization_type == NormalizationType.PCEN:
        return pcen_normalization(audio_data, *args)

    raise ValueError("Unsupported normalization type")
