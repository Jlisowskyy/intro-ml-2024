"""
audio_data.py

This module defines the AudioData class, which represents audio signals 
along with their sample rates. It is used in audio processing pipelines.
"""

import numpy as np


class AudioData:
    """
    A class to represent audio data.

    Attributes:
    audio_signal (np.array): The raw audio signal.
    sample_rate (int): The sample rate of the audio signal in Hertz.
    """

    def __init__(self, audio_signal: np.array, sample_rate: int):
        """
        Initializes the AudioData object with an audio signal and its sample rate.
        Converts the audio signal to a float32 array if not already in float format.

        Parameters:
        audio_signal (np.array): The audio signal as a numpy array.
        sample_rate (int): The sample rate of the audio signal in Hertz.
        """
        self.audio_signal = self.to_float(audio_signal)
        self.sample_rate = sample_rate

    @staticmethod
    def to_float(audio_data: np.array) -> np.array:
        """
        Converts audio data to a float32 array while ensuring that the audio
        representation is preserved.
        Only converts the data type if it is not already a float32 or float64 array.

        Parameters:
        audio_data (np.array): The audio data to convert to float32.

        Returns:
        np.array: The audio data as a float32 array.
        """
        if audio_data.dtype not in (np.float32, np.float64):
            dtype_max = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32)
            audio_data /= dtype_max
        return audio_data
