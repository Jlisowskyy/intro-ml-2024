"""
This module defines the AudioNormalizer class, which normalizes audio data
to a specified range. It is used in audio processing pipelines to prepare
audio signals for further analysis or modeling.
"""

from src.audio.audio_data import AudioData
from src.audio.normalize import normalize
from src.constants import NormalizationType


class AudioNormalizer:
    """
    A class to normalize audio data as part of a machine learning pipeline.
    """

    normalization_type = NormalizationType.CMVN
    def __init__(self, normalization_type: NormalizationType = NormalizationType.CMVN) -> None:
        """
        Initializes the AudioNormalizer.

        Parameters:
        normalization_type (NormalizationType): Type of normalization to perform.

        Returns:
        None
        """
        self.normalization_type = normalization_type
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
        for audio_data in x_data:
            audio_data.audio_signal = normalize(
                audio_data.audio_signal,
                audio_data.sample_rate,
                self.normalization_type)
        return x_data
