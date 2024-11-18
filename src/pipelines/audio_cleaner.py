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
from src.constants import DENOISE_FREQ_HIGH_CUT, DENOISE_FREQ_LOW_CUT, DenoiseType

class AudioCleaner:
    """
    A class used to clean audio data as part of a machine learning pipeline.
    """

    denoise_type = DenoiseType.BASIC
    def __init__(self, denoise_type: DenoiseType = DenoiseType.BASIC) -> None:
        """
        Initializes the AudioCleaner.

        Parameters:
        denoise_type (DenoiseType): Type of denoising to perform (from DenoiseType enum).

        Returns:
        None
        """
        self.denoise_type = denoise_type

    @staticmethod
    def denoise(chunk: np.ndarray, fs: float,
        denoise_type: DenoiseType = DenoiseType.BASIC) -> np.ndarray:
        """
        Denoise the given audio chunk using the specified denoise type.

        :param chunk: Audio chunk (numpy array) to be denoised
        :param fs: Sampling rate (frame rate in Hz)
        :param denoise_type: Type of denoising to perform
        :return: Denoised chunk of audio data
        """

        assert chunk.dtype in (np.float32, np.float64)
        if denoise_type == DenoiseType.BASIC:
            return AudioCleaner.denoise_basic(chunk, fs)

        raise ValueError(f"Unsupported denoise type: {denoise_type}")

    @staticmethod
    def denoise_basic(chunk: np.ndarray, fs: float) -> np.ndarray:
        """
        Perform basic denoising by applying a bandpass filter to the chunk of audio data.

        :param chunk: Audio chunk (numpy array) to be denoised
        :param fs: Sampling rate (frame rate in Hz)
        :return: Filtered chunk of audio data
        """

        sos = AudioCleaner.butter_bandpass(DENOISE_FREQ_LOW_CUT, DENOISE_FREQ_HIGH_CUT, fs)
        filtered_chunk = sosfilt(sos, chunk)

        return filtered_chunk

    @staticmethod
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
            audio_data.audio_signal = AudioCleaner.denoise(
                audio_data.audio_signal,
                audio_data.sample_rate,
                self.denoise_type)

        return x_data
