"""
This module defines a function to process audio data by applying various transformations
in sequence using scikit-learn's Pipeline. The transformations include denoising,
normalization, and generating a mel spectrogram.
"""

import numpy as np
from sklearn.pipeline import Pipeline

from src.constants import NORMALIZATION_TYPE, NormalizationType
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.pipeline.spectrogram_generator import SpectrogramGenerator


def process_audio(audio_data: AudioData,
                  normalization_type: NormalizationType = NORMALIZATION_TYPE) -> np.ndarray | None:
    """
    Process the audio data by denoising, normalizing, and generating a mel spectrogram.

    This function constructs a processing pipeline using sklearn's Pipeline, which applies
    the following transformations to the input audio data in sequence:
    1. **AudioCleaner**: Denoises the audio.
    2. **AudioNormalizer**: Normalizes the audio signal based on the specified normalization type.
    3. **SpectrogramGenerator**: Generates a mel-frequency spectrogram from the processed audio
        data.

    Args:
        audio_data (AudioData): The audio data to be processed, represented as an AudioData object.
        normalization_type (str): The type of normalization to apply. This value is passed to the
        AudioNormalizer.
        The default is NORMALIZATION_TYPE from src.constants.

    Returns:
        np.ndarray: The generated mel-frequency spectrogram, represented as a numpy array.
    """
    # Create a preprocessing pipeline with multiple steps
    preprocess_pipeline = Pipeline(steps=[
        ('AudioCleaner', AudioCleaner()),
        ('AudioNormalizer', AudioNormalizer(normalization_type=normalization_type)),
        ('SpectrogramGenerator', SpectrogramGenerator())
    ])

    # Transform the audio data using the pipeline
    transformed_data = preprocess_pipeline.transform([audio_data])

    # Return the processed mel spectrogram
    return transformed_data[0] if len(transformed_data) > 0 else None
