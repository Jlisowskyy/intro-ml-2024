"""
Author: Łukasz Kryczka

Test the fit_to_window function in the AudioNormalizer class.
"""
import numpy as np

from src.constants import (DEFAULT_TEST_FILES,
                           DEFAULT_SAVE_AUDIO,
                           DEFAULT_SAVE_SPECTROGRAMS,
                           DEFAULT_SHOULD_PLOT)
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_normalizer import AudioNormalizer
from src.test.test_transformation import test_transformation

TEST_FILES = DEFAULT_TEST_FILES


def fit_to_window_test() -> None:
    """
    Run the manual test for the silence removal module.
    Displays spectrograms before and after silence removal and saves processed files.
    """
    def transformation_func(x: np.ndarray, sr: int) -> np.ndarray:
        audio_normalizer = AudioNormalizer()
        current_window_length_seconds = x.shape[0] / sr
        audio_cleaner = AudioCleaner()
        x = audio_cleaner.denoise_raw(x, sr)
        x = audio_cleaner.remove_silence_raw(x, sr)
        return audio_normalizer.fit_to_window_raw(x, sr, current_window_length_seconds)

    test_transformation(transformation_func,
                        "remove_silence_then_fit_to_window",
                        TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)
