"""
Author: Łukasz Kryczka, 2024

This module provides functionality for denoising WAV data using simple filters.
Currently, supports basic denoising for human speech frequencies.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from src.audio.audio_data import AudioData

from src.constants import DENOISE_FREQ_HIGH_CUT, DENOISE_FREQ_LOW_CUT, DENOISE_NYQUIST_COEFFICIENT


def denoise(audio_data: AudioData,
            lowcut: float = DENOISE_FREQ_LOW_CUT,
            highcut: float = DENOISE_FREQ_HIGH_CUT) -> AudioData:
    """
    Denoise the given audio chunk using the specified denoise type.

    :param audio_data: Audio data to be denoised
    :param lowcut: Lower bound of the frequency range (Hz)
    :param highcut: Upper bound of the frequency range (Hz)

    :return: Denoised chunk of audio data
    """

    assert audio_data.audio_signal.dtype in (np.float32, np.float64)

    return butter_bandpass(audio_data, lowcut, highcut)


def butter_bandpass(audio_data: AudioData, lowcut: float, highcut: float,
                    order: int = 6) -> AudioData:
    """
    Create a bandpass filter to allow frequencies within a specified range and block others.

    :param audio_data: Audio data to be filtered
    :param lowcut: Lower bound of the frequency range (Hz)
    :param highcut: Upper bound of the frequency range (Hz)
    :param order: Order of the filter
    :return: Filtered audio data
    """

    nyquist = DENOISE_NYQUIST_COEFFICIENT * audio_data.sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError(
            f"Invalid critical frequencies: low={low}, high={high}. Ensure 0 < low < high < 1.")

    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_signal = sosfilt(sos, audio_data.audio_signal)
    audio_data.audio_signal = filtered_signal

    return audio_data
