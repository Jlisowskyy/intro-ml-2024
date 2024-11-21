"""
Author: Micał Kwiatkowski

This module processes audio data through a pipeline that cleans, normalizes, and generates mel 
spectrograms. The pipeline uses `AudioCleaner`, `AudioNormalizer`, and `SpectrogramGenerator` 
to transform the audio and save spectrograms as images.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.pipeline import Pipeline

from src.pipeline.audio_cleaner import AudioCleaner
# pylint: disable=line-too-long
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.pipeline.spectrogram_generator import SpectrogramGenerator

TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f2_script1_ipad_office1_35000.wav')))
SPECTROGRAM_CLEANED_PATH = str(Path.resolve(Path(f'{__file__}/../test_tmp/cleaned_spectrogram.png')))
SPECTROGRAM_PATH = str(Path.resolve(Path(f'{__file__}/../test_tmp/spectrogram.png')))


def example_test_run():
    """
    Run an example of the transformation pipeline on the first audio file in the dataset.
    
    The pipeline includes cleaning, normalizing, and generating spectrograms. It saves both
    the cleaned and uncleaned spectrograms.
    """
    transformation_pipeline = Pipeline(steps=[
        ('AudioCleaner', AudioCleaner()),
        ('AudioNormalizer', AudioNormalizer()),
        ('SpectrogramGenerator', SpectrogramGenerator())
    ])

    audio_data_wav, sample_rate = sf.read(TEST_FILE_PATH)
    audio_data = AudioData(np.array(audio_data_wav), sample_rate)

    # Transformed data
    transformation_pipeline.fit([audio_data])
    model_input = transformation_pipeline.transform([audio_data])

    SpectrogramGenerator.save_spectrogram(model_input[0], SPECTROGRAM_CLEANED_PATH)

    spectrogram_not_cleaned = SpectrogramGenerator.gen_spectrogram(audio_data, mel=True)
    SpectrogramGenerator.save_spectrogram(spectrogram_not_cleaned, SPECTROGRAM_PATH)

