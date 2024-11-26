"""
Author: Jakub Lisowski, Łukasz Kryczka

File tests quality of silence removal
"""

from src.constants import (DEFAULT_TEST_FILES,
                           DEFAULT_SAVE_AUDIO, DEFAULT_SAVE_SPECTROGRAMS, DEFAULT_SHOULD_PLOT)
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.test.test_transformation import test_transformation

TEST_FILES = DEFAULT_TEST_FILES


def silence_removal_test() -> None:
    """
    Run the manual test for the silence removal module.
    Displays spectrogram before and after silence removal and saves processed files.
    """

    def transform_func(audio_data: AudioData) -> AudioData:
        cleaner = AudioCleaner()
        audio_data = cleaner.denoise(audio_data)
        audio_data = cleaner.remove_silence(audio_data)
        return audio_data

    test_transformation(transform_func, "silence_removal", TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)
