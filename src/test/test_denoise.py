"""
Author: Łukasz Kryczka

Manual test cases for denoise module using a pretrained DNS64 model.
"""

import numpy as np

from src.constants import (DEFAULT_TEST_FILES,
                           DEFAULT_SAVE_AUDIO,
                           DEFAULT_SAVE_SPECTROGRAMS,
                           DEFAULT_SHOULD_PLOT)
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.test.test_transformation import test_transformation

TEST_FILES = DEFAULT_TEST_FILES


def denoise_test_manual():
    """
    Run the manual test for the denoise module.
    It loads a WAV file, adds noise to the audio signal, denoises it using the
    AudioCleaner class, and displays the original and denoised spectrograms.

    The 'original + noise' and 'denoised' audio signals are saved to WAV files.
    """

    def preprocess_func(audio_data: AudioData) -> AudioData:
        noise = np.random.normal(0, 0.02, audio_data.audio_signal.shape)
        audio_data.audio_signal += noise
        return audio_data

    def transformation_func(audio_data: AudioData) -> AudioData:
        audio_cleaner = AudioCleaner()
        return audio_cleaner.denoise(audio_data)

    test_transformation(transformation_func,
                        "denoise",
                        TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT,
                        preprocess_func=preprocess_func)
