"""
Author: Michał Kwiatkowski, Łukasz Kryczka

This module processes audio data through a pipeline that cleans, normalizes, and generates mel 
spectrograms. The pipeline uses `AudioCleaner`, `AudioNormalizer`, and `SpectrogramGenerator` 
to transform the audio and save spectrograms as images.
"""

from sklearn.pipeline import Pipeline

from src.constants import (DEFAULT_TEST_FILES,
                           DEFAULT_SAVE_AUDIO,
                           DEFAULT_SAVE_SPECTROGRAMS,
                           DEFAULT_SHOULD_PLOT)
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.test.test_transformation import test_transformation

TEST_FILES = DEFAULT_TEST_FILES


def transformation_pipeline_test() -> None:
    """
    Manual test for the whole transformation pipeline
    """
    def transformation_func(audio_data: AudioData) -> AudioData:
        transformation_pipeline = Pipeline(steps=[
            ('AudioCleaner', AudioCleaner()),
            ('AudioNormalizer', AudioNormalizer()),
        ])
        transformation_pipeline.fit([audio_data])
        return transformation_pipeline.transform([audio_data])[0]

    test_transformation(transformation_func,
                        "transformation_pipeline",
                        TEST_FILES,
                        save_audio=DEFAULT_SAVE_AUDIO,
                        save_spectrograms=DEFAULT_SAVE_SPECTROGRAMS,
                        plot=DEFAULT_SHOULD_PLOT)
