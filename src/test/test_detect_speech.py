"""
Author: Jakub Lisowski

File tests quality of silence removal
"""
import numpy as np

from src.constants import (DEFAULT_TEST_FILES,
                           DEFAULT_SAVE_AUDIO, DEFAULT_SAVE_SPECTROGRAMS, DEFAULT_SHOULD_PLOT)
from src.pipeline.audio_cleaner import AudioCleaner
from src.test.test_transformation import test_transformation

TEST_FILES = DEFAULT_TEST_FILES


def silence_removal_test() -> None:
    """
    Run the manual test for the silence removal module.
    Displays spectrograms before and after silence removal and saves processed files.
    """
    def transform_func(x: np.ndarray, sr: int) -> np.ndarray:
        cleaner = AudioCleaner()
        denoised = cleaner.denoise_raw(x, sr)
        cleaned = cleaner.remove_silence_raw(denoised, sr)
        return cleaned

    test_transformation(transform_func, "silence_removal", TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)
