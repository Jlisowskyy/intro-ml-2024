"""
Author: Michał Kwiatkowski, Tomasz Mycielski

Module for classifying audio data using a CNN model.

This module contains the classify function which processes audio data
through a series of transformations and passes it to a CNN model for prediction.
"""

from sklearn.pipeline import Pipeline
import torch

from src.audio.audio_data import AudioData
from src.cnn.cnn import BasicCNN
from src.pipelines.audio_cleaner import AudioCleaner
from src.pipelines.audio_normalizer import AudioNormalizer
from src.pipelines.classifier import Classifier
from src.pipelines.spectrogram_generator import SpectrogramGenerator
from src.pipelines.tensor_transform import TensorTransform


def classify(audio_data: list[AudioData], model: BasicCNN) -> list[int]:
    """
    Classify audio data using the provided CNN model.

    Args:
        data (AudioData): The audio data to classify.
        model (BasicCNN): The CNN model used for classification.

    Returns:
        int: user's class.
    """

    transformation_pipeline = Pipeline(steps=[
        ('AudioCleaner', AudioCleaner()),
        ('AudioNormalizer', AudioNormalizer()),
        ('SpectrogramGenerator', SpectrogramGenerator()),
        ('TensorTransform', TensorTransform()),
        ('Classifier', Classifier(model))
    ])

    # Transformed data
    transformation_pipeline.fit([audio_data])
    predictions = transformation_pipeline.predict([audio_data])

    return predictions
