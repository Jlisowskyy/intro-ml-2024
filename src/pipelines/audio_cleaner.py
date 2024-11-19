"""
Author: Michał Kwiatkowski

This module contains the AudioCleaner class, which is used to clean audio data
as part of a machine learning pipeline. It implements the fit and transform
methods to be compatible with scikit-learn pipelines.
"""

from src.audio.audio_data import AudioData
from src.audio.denoise import denoise


class AudioCleaner:
    """
    A class used to clean audio data as part of a machine learning pipeline.
    """

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
            audio_data.audio_signal = denoise(
                audio_data.audio_signal,
                audio_data.sample_rate
            )

        return x_data
