"""
Author: Michał Kwiatkowski

Module for classifying audio data using a CNN model.

This module contains the classify function which processes audio data
through a series of transformations and passes it to a CNN model for prediction.
"""

import numpy as np
from sklearn.pipeline import Pipeline

from src.cnn.cnn import TutorialCNN
from src.pipelines.audio_cleaner import AudioCleaner
from src.pipelines.audio_data import AudioData
from src.pipelines.audio_normalizer import AudioNormalizer
from src.pipelines.spectrogram_generator import SpectrogramGenerator

def classify(data: np.array, sample_rate: int, model: TutorialCNN) -> np.array:
    """
    Classify audio data using the provided CNN model.

    Args:
        data (np.array): The audio data to classify.
        sample_rate (int): The sample rate of the audio data.
        model (TutorialCNN): The CNN model used for classification.

    Returns:
        np.array: The predictions made by the model.
    """

    audio_data = AudioData(data, sample_rate)

    transformation_pipeline = Pipeline(steps=[
        ('AudioCleaner', AudioCleaner()),
        ('AudioNormalizer', AudioNormalizer()),
        ('SpectrogramGenerator', SpectrogramGenerator())
    ])

    # Transformed data
    transformation_pipeline.fit([audio_data])
    model_input = transformation_pipeline.transform([audio_data])

    prediction = model(model_input)
    return prediction
