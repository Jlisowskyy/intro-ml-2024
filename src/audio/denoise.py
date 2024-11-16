"""
Author: Łukasz Kryczka, 2024

This module provides functionality for denoising WAV data using simple filters.
Currently, supports basic denoising for human speech frequencies.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from src.constants import DENOISE_FREQ_HIGH_CUT, DENOISE_FREQ_LOW_CUT


def denoise(chunk: np.array,
            fs: float,
            lowcut: float = DENOISE_FREQ_LOW_CUT,
            highcut: float = DENOISE_FREQ_HIGH_CUT) -> np.array:
    """
    Denoise the given audio chunk using the specified denoise type.

    :param chunk: Audio chunk (numpy array) to be denoised
    :param fs: Sampling rate (frame rate in Hz)
    :param lowcut: Lower bound of the frequency range (Hz)
    :param highcut: Upper bound of the frequency range (Hz)

    :return: Denoised chunk of audio data
    """

    assert chunk.dtype in (np.float32, np.float64)

    return butter_bandpass(chunk, lowcut, highcut, fs)


def butter_bandpass(chunk: np.array, lowcut: float, highcut: float, fs: float,
                    order: int = 6) -> np.array:
    """
    Create a bandpass filter to allow frequencies within a specified range and block others.

    :param chunk: Audio chunk (numpy array) to be denoised
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
    filtered_chunk = sosfilt(sos, chunk)

    return filtered_chunk
