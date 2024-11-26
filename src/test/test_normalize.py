"""
Author: Łukasz Kryczka

Test cases for the normalization module.
Tests the mean-variance normalization and PCEN using WAV file input.
"""

from src.constants import (DEFAULT_TEST_FILES,
                           DEFAULT_SAVE_AUDIO,
                           DEFAULT_SAVE_SPECTROGRAMS,
                           DEFAULT_SHOULD_PLOT)
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.test.test_transformation import test_transformation

TEST_FILES = DEFAULT_TEST_FILES


def mean_variance_normalization_manual_test() -> None:
    """
    Manual test for the mean-variance normalization
    """
    def transformation_func(audio_data: AudioData) -> AudioData:
        audio_normalizer = AudioNormalizer()
        return audio_normalizer.mean_variance_normalization(audio_data)

    test_transformation(transformation_func,
                        "mean_variance_normalization",
                        TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)


def pcen_normalization_manual_test() -> None:
    """
    Manual test for the PCEN normalization
    """
    def transformation_func(audio_data: AudioData) -> AudioData:
        audio_normalizer = AudioNormalizer()
        return audio_normalizer.pcen_normalization(audio_data)

    test_transformation(transformation_func,
                        "pcen_normalization",
                        TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)
