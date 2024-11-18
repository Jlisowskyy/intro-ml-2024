"""
Author: Michał Kwiatkowski, Łukasz Kryczka

This module defines the AudioNormalizer class, which normalizes audio data
to a specified range. It is used in audio processing pipelines to prepare
audio signals for further analysis or modeling.
"""

import librosa
import numpy as np
from src.audio.audio_data import AudioData
from src.constants import NormalizationType

class AudioNormalizer:
    """
    A class to normalize audio data as part of a machine learning pipeline.
    """

    def __init__(self,
                 normalization_type: NormalizationType = NormalizationType.MEAN_VARIANCE) -> None:
        """
        Initializes the AudioNormalizer.

        Parameters:
        normalization_type (NormalizationType): Type of normalization to perform.

        Returns:
        None
        """
        self.normalization_type = normalization_type

    @staticmethod
    def mean_variance_normalization(signal: np.array) -> np.array:
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

    @staticmethod
    def cmvn_normalization(signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Apply Cepstral Mean and Variance Normalization (CMVN) to the signal.

        Source: https://en.wikipedia.org/wiki/Cepstral_mean_and_variance_normalization

        :param signal: Input audio signal (numpy array)
        :param fs: Sampling rate
        :return: CMVN normalized audio signal (numpy array)
        """

        raise NotImplementedError("CMVN normalization is not implemented yet")

    @staticmethod
    def pcen_normalization(signal: np.ndarray,
                        fs: float,
                        time_constant: float = 0.06,
                        alpha: float = 0.98,
                        delta: float = 2,
                        r: float = 0.5,
                        eps: float = 1e-6) -> np.ndarray:
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
        m = librosa.pcen(s, sr=fs, hop_length=512,
                        gain=alpha, bias=delta,
                        eps=eps, power=r,
                        time_constant=time_constant)
        normalized_signal = librosa.istft(m)
        return normalized_signal

    @staticmethod
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
            return AudioNormalizer.mean_variance_normalization(signal)
        if normalization_type == NormalizationType.PCEN:
            return AudioNormalizer.pcen_normalization(signal, fs, *args)
        if normalization_type == NormalizationType.CMVN:
            return AudioNormalizer.cmvn_normalization(signal, fs)

        raise ValueError("Unsupported normalization type")

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None):
        """
        Fits the normalizer to the audio data (no fitting process required).

        Parameters:
        x_data (list[AudioData]): A list of AudioData instances to fit on.
        y_data (list[int], optional): Target values (ignored in this transformer).

        Returns:
        self: Returns the instance itself.
        """
        # No fitting process required for normalization
        return self

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Transforms the input audio data by normalizing the audio signals.

        Parameters:
        x_data (list[AudioData]): A list of AudioData instances to normalize.

        Returns:
        list[AudioData]: A list of normalized AudioData instances.
        """
        for audio_data in x_data:
            audio_data.audio_signal = AudioNormalizer.normalize(
                audio_data.audio_signal,
                audio_data.sample_rate,
                self.normalization_type)
        return x_data
