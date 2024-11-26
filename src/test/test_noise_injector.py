"""
Author: Michał Kwiatkowski
"""

from src.constants import (DEFAULT_FILE_NAMES,
                           DEFAULT_SAVE_AUDIO,
                           DEFAULT_SAVE_SPECTROGRAMS,
                           DEFAULT_SHOULD_PLOT, TEST_FOLDER_IN, TEST_FOLDER_OUT)
from src.pipeline.audio_data import AudioData
from src.pipeline.noise_injector import NoiseInjector
from src.test.test_file import TestFile
from src.test.test_transformation import test_transformation

TEST_FILES = {TestFile(
        str(TEST_FOLDER_IN / DEFAULT_FILE_NAMES[2]),
        DEFAULT_FILE_NAMES[2],
        str(TEST_FOLDER_OUT / DEFAULT_FILE_NAMES[2])
    )}


def noise_injector_test() -> None:
    """
    """
    def transformation_func(audio_data: AudioData) -> AudioData:
        noise_injector = NoiseInjector()
        audio_with_noise = noise_injector.transform([audio_data])
        return audio_with_noise[0]

    test_transformation(transformation_func,
                        "add_noise",
                        TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)
