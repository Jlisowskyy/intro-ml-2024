"""
Author: Michał Kwiatkowski, Łukasz Kryczka

This module contains the AudioCleaner class, which is used to clean audio data
as part of a machine learning pipeline. It implements the fit and transform
methods to be compatible with scikit-learn pipelines. It provides
functionality for denoising WAV data using simple filters. Currently, supports
basic denoising for human speech frequencies.
"""

from scipy.signal import butter, sosfilt
import numpy as np
from src.audio.audio_data import AudioData
from src.constants import DENOISE_FREQ_HIGH_CUT, DENOISE_FREQ_LOW_CUT, DENOISE_NYQUIST_COEFFICIENT

class AudioCleaner:
    """
    A class used to clean audio data as part of a machine learning pipeline.
    """
    def __init__(self) -> None:
        """
        Initializes the AudioCleaner.

        Parameters:
        denoise_type (DenoiseType): Type of denoising to perform (from DenoiseType enum).

        Returns:
        None
        """
        return

    @staticmethod
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

        # assert audio_data.audio_signal.dtype in (np.float32, np.float64)

        return AudioCleaner.butter_bandpass(audio_data, lowcut, highcut)

    @staticmethod
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

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None):
        """
        Fits the transformer to the data.

        Parameters:
        x_data (array-like): Input data - list of AudioData objects.
        y_data (array-like, optional): Target values (ignored in this transformer).

        Returns:
        self: Returns the instance itself.
        """
        # No fitting process for this cleaner, but required for sklearn pipeline compatibility
        return self

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Transforms the input data by cleaning the audio.

        Parameters:
        x_data (array-like): Input data - list of AudioData objects.

        Returns:
        x_data (array-like): Output data - list of cleaned AudioData objects.
        """
        for audio_data in x_data:
            audio_data = AudioCleaner.denoise(
                audio_data
            )

        return x_data
