"""
Author: Łukasz Kryczka, 2024

This module provides functionality for denoising WAV data using simple filters.
Currently, supports basic denoising for human speech frequencies.
"""

from enum import Enum
import numpy as np
from scipy.signal import butter, sosfilt


class DenoiseType(Enum):
    """
    Enum for different types of denoising.
    Future types of denoising can be added here and
    handled in the denoise function
    """
    BASIC = 1


def denoise(chunk: np.ndarray,
            fs: float,
            denoise_type: DenoiseType = DenoiseType.BASIC) -> np.ndarray:
    """
    Denoise the given audio chunk using the specified denoise type.

    :param chunk: Audio chunk (numpy array) to be denoised
    :param fs: Sampling rate (frame rate in Hz)
    :param denoise_type: Type of denoising to perform
    :return: Denoised chunk of audio data
    """
    if denoise_type == DenoiseType.BASIC:
        return denoise_basic(chunk, fs)

    raise ValueError(f"Unsupported denoise type: {denoise_type}")


def butter_bandpass(lowcut, highcut, fs, order=18):
    """
    Create a bandpass filter to allow frequencies within a specified range and block others.

    :param lowcut: Lower bound of the frequency range (Hz)
    :param highcut: Upper bound of the frequency range (Hz)
    :param fs: Sampling rate (frame rate in Hz)
    :param order: Order of the filter
    :return: Second-order sections for the bandpass filter
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def denoise_basic(chunk: np.ndarray, fs: float) -> np.ndarray:
    """
    Perform basic denoising by applying a bandpass filter to the chunk of audio data.

    :param chunk: Audio chunk (numpy array) to be denoised
    :param fs: Sampling rate (frame rate in Hz)
    :return: Filtered chunk of audio data
    """

    lowcut = 80.0  # Lower bound (Male voice frequency range)
    highcut = 8200.0  # Upper bound (common male and female voices frequency range)

    sos = butter_bandpass(lowcut, highcut, fs)
    filtered_chunk = sosfilt(sos, chunk)

    return filtered_chunk
