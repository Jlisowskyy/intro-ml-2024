"""
Author: Michał Kwiatkowski, Łukasz Kryczka

This module defines the AudioNormalizer class, which normalizes audio data
to a specified range. It is used in audio processing pipelines to prepare
audio signals for further analysis or modeling.
"""

import librosa
import numpy as np

from src.constants import (NormalizationType, EPSILON, NORMALIZATION_PCEN_TIME_CONSTANT,
                           NORMALIZATION_PCEN_ALPHA, NORMALIZATION_PCEN_DELTA,
                           NORMALIZATION_PCEN_R, NORMALIZATION_PCEN_HOP_LENGTH,
                           DETECT_SILENCE_THRESHOLD_DB, DETECT_SILENCE_WINDOW_MAX_MS,
                           SILENCE_CUT_WINDOW_MS, MODEL_WINDOW_LENGTH)
from src.pipeline.audio_data import AudioData


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
    def fit_to_window(audio_data: AudioData, window_length_seconds: float) -> AudioData:
        """
        Fit the audio data to a specified window length by repeating the signal
        Raises an error if the window length is smaller than the signal length

        :param audio_data: Audio data to be fitted
        :param window_length_seconds: Length of the window to fit the signal to
        """
        audio_data.audio_signal = AudioNormalizer.fit_to_window_raw(audio_data.audio_signal,
                                                                    audio_data.sample_rate,
                                                                    window_length_seconds)
        return audio_data

    @staticmethod
    def fit_to_window_raw(x: np.ndarray, sr: int, window_length_seconds: float) -> np.ndarray:
        """
        Fit the audio data to a specified window length by repeating the signal
        Raises an error if the window length is smaller than the signal length

        :param x: Audio data to be fitted
        :param sr: Sample rate of the audio data
        :param window_length_seconds: Length of the window to fit the signal to
        """
        window_size = int(window_length_seconds * sr)
        if window_size < len(x):
            raise ValueError("Window length is smaller than the signal length")

        num_repeats = window_size // len(x) + 1
        reps = (num_repeats,) + (1,) * (x.ndim - 1)
        x = np.tile(x, reps)
        x = x[:window_size]
        return x

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
            return AudioNormalizer.mean_variance_normalization(audio_data)
        if normalization_type == NormalizationType.PCEN:
            return AudioNormalizer.pcen_normalization(audio_data, *args)

        raise ValueError("Unsupported normalization type")

    @staticmethod
    def is_speech(audio_data: AudioData, silence_threshold=DETECT_SILENCE_THRESHOLD_DB) -> bool:
        """
        Determines if an audio segment contains speech based on RMS energy threshold.

        Parameters:
            audio_data (AudioData): Audio segment to analyze.
            silence_threshold (float): RMS energy threshold in dB to classify as speech.

        Returns:
            bool: True if segment contains speech, False if classified as silence.
        """
        duration = len(audio_data.audio_signal) / audio_data.sample_rate
        assert (duration * 1000) < DETECT_SILENCE_WINDOW_MAX_MS

        rms = np.sqrt(np.mean(np.square(audio_data.audio_signal)))
        rms_db = 20 * np.log10(rms + 1e-9)

        return rms_db > silence_threshold

    @staticmethod
    def remove_silence_raw(audio_data: np.ndarray, frame_rate: int,
                           silence_threshold: int = DETECT_SILENCE_THRESHOLD_DB) -> np.ndarray:
        """
        Removes silent segments from raw audio data by analyzing fixed-size windows.
        Expects and returns audio data in shape (N, 1).

        Parameters:
            audio_data (np.ndarray): Raw audio signal data in shape (N, 1).
            frame_rate (int): Audio sample rate in Hz.
            silence_threshold (int): RMS energy threshold in dB to classify as speech.

        Returns:
            np.ndarray: Audio data with silent segments removed, maintaining shape (N, 1).

        Raises:
            ValueError: If the input audio is not in shape (N, 1).
        """

        if len(audio_data.shape) != 2 or audio_data.shape[1] != 1:
            raise ValueError(f"Input audio must be in shape (N, 1). Got shape: {audio_data.shape}")

        audio_1d = audio_data.flatten()
        audio = AudioData(audio_1d, frame_rate)
        processed_audio = AudioNormalizer.remove_silence(audio, silence_threshold)
        return processed_audio.audio_signal.reshape(-1, 1)

    @staticmethod
    def remove_silence(audio_data: AudioData,
                       silence_threshold: int = DETECT_SILENCE_THRESHOLD_DB) -> AudioData:
        """
        Removes silent segments from an AudioData object.

        Parameters:
            audio_data (AudioData): Audio data to process, with audio_signal as numpy.ndarray.
            silence_threshold (float): RMS energy threshold in dB to classify as speech.

        Returns:
            AudioData: New AudioData object with silent segments removed,
             maintaining input dimensions.
        """
        window = (SILENCE_CUT_WINDOW_MS * audio_data.sample_rate) // 1000

        output = np.array([], dtype=audio_data.audio_signal.dtype).reshape(
            (-1,) + audio_data.audio_signal.shape[1:])

        for i in range(0, len(audio_data.audio_signal), window):
            chunk_data = audio_data.audio_signal[i:i + window]
            chunk = AudioData(chunk_data, audio_data.sample_rate)

            if AudioNormalizer.is_speech(chunk, silence_threshold):
                if len(chunk_data.shape) < len(output.shape):
                    chunk_data = chunk_data.reshape((-1,) + output.shape[1:])
                output = np.concatenate((output, chunk_data), axis=0)

        return AudioData(output, audio_data.sample_rate)

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'AudioNormalizer':
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
        transformed_data = []
        for audio_data in x_data:
            normalized = AudioNormalizer.normalize(audio_data, self.normalization_type)
            silence_removed = AudioNormalizer.remove_silence(normalized)

            if len(silence_removed.audio_signal) == 0:
                continue

            fitted = AudioNormalizer.fit_to_window(silence_removed, MODEL_WINDOW_LENGTH)
            transformed_data.append(fitted)
        return transformed_data
