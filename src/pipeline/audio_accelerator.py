"""
Author: Michał Kwiatkowski

Provides the `AudioAccelerator` class for augmenting audio data by altering its speed.
Designed for use in audio data pipelines with adjustable speed_factor parameter.
"""

from src.constants import AUDIO_AUGMENTATION_DEFAULT_SPEED_FACTOR
from src.pipeline.audio_data import AudioData
from src.scripts.audio_augmentation import change_speed

class AudioAccelerator:
    """
    A class for augmenting audio data by adjusting its speed (tempo).

    Attributes:
        speed_factor (float): A factor by which the audio speed will be adjusted.
    """

    def __init__(self, speed_factor: float = AUDIO_AUGMENTATION_DEFAULT_SPEED_FACTOR) -> None:
        """
        Initializes an AudioAccelerator instance with the specified speed adjustment factor.

        Parameters:
            speed_factor (float, optional): The factor by which the speed of audio is adjusted.
                                             Defaults to AUDIO_AUGMENTATION_DEFAULT_SPEED_FACTOR.
        """
        self.speed_factor = speed_factor

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'AudioAccelerator':
        """
        A placeholder method for fitting the transformer. 
        This function does not perform any actions but is included for consistency 
        with scikit-learn's transformer API.

        Parameters:
            x_data (list[AudioData]): List of input audio data to fit.
            y_data (list[int], optional): List of target labels corresponding to
                                        the input audio data. Defaults to None.

        Returns:
            AudioAccelerator: The instance of the AudioAccelerator class (unchanged).
        """
        return self

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Adjusts the speed of a list of audio data objects based on the specified speed factor.

        Parameters:
            x_data: list[AudioData]: List of AudioData objects to transform by altering their speed.

        Returns:
            list[AudioData]: A list of AudioData objects with modified speed.
        """
        accelerated_audio = []

        for audio_data in x_data:
            accelerated_audio.append(change_speed(audio_data, self.speed_factor))

        return accelerated_audio
