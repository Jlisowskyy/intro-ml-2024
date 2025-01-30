"""
Author: Łukasz Kryczka

Test the fit_to_window function in the AudioNormalizer class.
"""

from src.constants import (DEFAULT_TEST_FILES,
                           DEFAULT_SAVE_AUDIO,
                           DEFAULT_SAVE_SPECTROGRAMS,
                           DEFAULT_SHOULD_PLOT)
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.test.test_transformation import test_transformation

TEST_FILES = DEFAULT_TEST_FILES


def fit_to_window_test() -> None:
    """
    Run the manual test for the silence removal module.
    Displays spectrograms before and after silence removal and saves processed files.
    """
    def transformation_func(audio_data: AudioData) -> AudioData:
        audio_normalizer = AudioNormalizer()
        audio_cleaner = AudioCleaner()
        current_window_length_seconds = audio_data.audio_signal.shape[0] / audio_data.sample_rate
        audio_data = audio_cleaner.denoise(audio_data)
        audio_data = AudioNormalizer.remove_silence(audio_data)
        return audio_normalizer.fit_to_window(audio_data, current_window_length_seconds)

    test_transformation(transformation_func,
                        "remove_silence_then_fit_to_window",
                        TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)
