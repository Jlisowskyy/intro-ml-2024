"""
Author: Michał Kwiatkowski

Provides the `EchoInjector` class for augmenting audio data by adding echo.
Designed for use in audio data pipelines with adjustable speed_factor parameter.
"""

from src.constants import AUDIO_AUGMENTATION_DEFAULT_ECHO_DELAY, \
    AUDIO_AUGMENTATION_DEFAULT_ECHO_DECAY

from src.pipeline.audio_data import AudioData
from src.scripts.audio_augmentation import add_echo

class EchoInjector:
    """
    A class for adding echo effects to audio data.

    This class applies an echo effect by manipulating the delay and decay of the audio.
    The echo effect can be customized using the `delay` and `decay` parameters.

    Attributes:
        delay (float): The delay time (in seconds) between the original audio signal 
                       and the echo. Larger values create longer delays.
        decay (float): The decay factor of the echo, controlling how quickly the echo fades.
                       Values closer to 1.0 result in a slower fade, while lower values 
                       create a faster decay.
    """
    def __init__(self, delay: float = AUDIO_AUGMENTATION_DEFAULT_ECHO_DELAY,
                 decay: float = AUDIO_AUGMENTATION_DEFAULT_ECHO_DECAY) -> None:
        """
        Initializes an AudioAccelerator instance with the specified speed adjustment factor.

        Parameters:
            speed_factor (float, optional): The factor by which the speed of audio is adjusted.
                                             Defaults to AUDIO_AUGMENTATION_DEFAULT_SPEED_FACTOR.
        """
        self.delay = delay
        self.decay = decay

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'EchoInjector':
        """
        A placeholder method for fitting the transformer. 
        This function does not perform any actions but is included for consistency 
        with scikit-learn's transformer API.

        Parameters:
            x_data (list[AudioData]): List of input audio data to fit.
            y_data (list[int], optional): List of target labels corresponding to
                                        the input audio data. Defaults to None.

        Returns:
            EchoInjector: The instance of the EchoInjector class (unchanged).
        """
        return self

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Adds echo to the AudioData from x_data.

        Parameters:
            x_data: list[AudioData]: List of AudioData objects to transform by adding echo.

        Returns:
            list[AudioData]: A list of AudioData objects with additional echo.
        """
        echo_audio = []

        for audio_data in x_data:
            echo_audio.append(add_echo(audio_data, self.delay, self.decay))

        return echo_audio
