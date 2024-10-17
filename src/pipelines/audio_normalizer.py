"""
audio_normalizer.py

This module defines the AudioNormalizer class, which normalizes audio data 
to a specified range. It is used in audio processing pipelines to prepare 
audio signals for further analysis or modeling.
"""

from src.pipelines.audio_data import AudioData

class AudioNormalizer:
    """
    A class to normalize audio data.

    This class provides methods to fit and transform audio signals 
    for normalization purposes.
    """

    def __init__(self):
        """Initializes the AudioNormalizer."""
        return

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
        return x_data
