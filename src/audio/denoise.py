"""
Author: Łukasz Kryczka, 2024

This module provides functionality for denoising WAV data using simple filters.
Currently, supports basic denoising for human speech frequencies.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from src.constants import DENOISE_FREQ_HIGH_CUT, DENOISE_FREQ_LOW_CUT, DenoiseType


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

    assert(chunk.dtype == np.float32 or chunk.dtype == np.float64)
    if denoise_type == DenoiseType.BASIC:
        return denoise_basic(chunk, fs)

    raise ValueError(f"Unsupported denoise type: {denoise_type}")


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 6) -> any:
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
    if low <= 0 or high >= 1:
        raise ValueError(
            f"Invalid critical frequencies: low={low}, high={high}. Ensure 0 < low < high < 1.")
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')

    return sos


def denoise_basic(chunk: np.ndarray, fs: float) -> np.ndarray:
    """
    Perform basic denoising by applying a bandpass filter to the chunk of audio data.

    :param chunk: Audio chunk (numpy array) to be denoised
    :param fs: Sampling rate (frame rate in Hz)
    :return: Filtered chunk of audio data
    """

    sos = butter_bandpass(DENOISE_FREQ_LOW_CUT, DENOISE_FREQ_HIGH_CUT, fs)
    filtered_chunk = sosfilt(sos, chunk)

    return filtered_chunk
