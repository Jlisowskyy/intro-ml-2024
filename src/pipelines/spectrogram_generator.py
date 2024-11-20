"""
Author: Michał Kwiatkowski

This module contains the SpectrogramGenerator class, which provides functionality
for generating mel-frequency spectrograms from audio data.
"""

import numpy as np

from src.audio.audio_data import AudioData
from src.audio.spectrogram import gen_spectrogram


class SpectrogramGenerator:
    """
    A class to generate mel-frequency spectrogram's from audio data.

    This class provides methods to fit the model (if applicable) and to transform
    audio data into spectrogram representations.
    """

    def __init__(self):
        """
        Initializes the SpectrogramGenerator instance.
        """
        return

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None):
        """
        Fit the generator to the audio data (if necessary).

        Args:
            x_data (list[AudioData]): A list of AudioData instances.
            y_data (list[int], optional): A list of labels (if applicable).

        Returns:
            self: Returns an instance of the fitted generator.
        """
        return self

    def transform(self, audio_data_list: list[AudioData]) -> list[np.ndarray]:
        """
        Transform audio data into mel-frequency spectrogram's.

        Args:
            audio_data_list (list[AudioData]): A list of AudioData instances 
            to be transformed into spectrogram's.

        Returns:
            list[np.ndarray]: A list of NumPy arrays representing the spectrogram's.
        """
        spectrogram_data = []
        for audio_data in audio_data_list:
            spectrogram = gen_spectrogram(audio_data, True)
            spectrogram_data.append(spectrogram)
        return spectrogram_data
