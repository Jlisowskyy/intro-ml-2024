"""
Author: Michał Kwiatkowski

Provides the `AudioPitcher` class for augmenting audio data by altering its pitch.
Designed for use in audio data pipelines with adjustable semitone parameters.
"""

from src.constants import AUDIO_AUGMENTATION_DEFAULT_SEMITONES
from src.pipeline.audio_data import AudioData
from src.scripts.audio_augmentation import change_pitch

class AudioPitcher:
    """
    A class for augmenting audio data by altering its pitch.

    Attributes:
        semitones (float): The number of semitones to adjust the pitch.
                           Positive values increase the pitch, negative values decrease it.
    """

    def __init__(self, semitones: float = AUDIO_AUGMENTATION_DEFAULT_SEMITONES) -> None:
        self.semitones = semitones

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'AudioPitcher':
        """
        Placeholder method for fitting, which is not implemented.
        This function does not perform any actions in the current version.

        Parameters:
            x_data (list[AudioData]): List of input audio data to fit.
            y_data (list[int], optional): List of target labels corresponding to the input audio
                                          data.

        Returns:
            AudioPitcher: The instance of the AudioPitcher class (unchanged).
        """
        return

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Applies pitch alteration to a list of audio data objects.

        Parameters:
            x_data (list[AudioData]): A list of AudioData objects to transform.

        Returns:
            list[AudioData]: A list of AudioData objects with adjusted pitch.
        """
        pitched_audio = []

        for audio_data in x_data:
            pitched_audio.append(change_pitch(audio_data, self.semitones))

        return pitched_audio
