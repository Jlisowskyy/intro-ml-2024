"""
audio_data.py

This module defines the AudioData class, which represents audio signals 
along with their sample rates. It is used in audio processing pipelines.
"""

import numpy as np

class AudioData:
    # pylint: disable=too-few-public-methods
    """
    A class to represent audio data.

    Attributes:
    audio_signal (np.array): The raw audio signal.
    sample_rate (int): The sample rate of the audio signal in Hertz.
    """
    def __init__(self, audio_signal: np.array, sample_rate: int):
        """
        Initializes the AudioData object with an audio signal and its sample rate.

        Parameters:
        audio_signal (np.array): The audio signal as a numpy array.
        sample_rate (int): The sample rate of the audio signal in Hertz.
        """
        self.audio_signal = audio_signal
        self.sample_rate = sample_rate
