"""
Author: Michał Kwiatkowski, Tomasz Mycielski

Module for classifying audio data using a CNN model.

This module contains the classify function which processes audio data
through a series of transformations and passes it to a CNN model for prediction.
"""

from sklearn.pipeline import Pipeline
import torch

from src.cnn.cnn import BasicCNN
from src.pipelines.load_model import load_model
from src.pipelines.audio_cleaner import AudioCleaner
from src.audio.audio_data import AudioData
from src.pipelines.audio_normalizer import AudioNormalizer
from src.pipelines.spectrogram_generator import SpectrogramGenerator
from src.pipelines.tensor_transform import TensorTransform

MODEL_PATH = ""
assert MODEL_PATH == "", "First setup model path"

def classify(audio_data: AudioData, model: BasicCNN) -> int:
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
        ('TensorTransform', TensorTransform())
    ])

    # Transformed data
    transformation_pipeline.fit([audio_data])
    model_input = transformation_pipeline.transform([audio_data])

    # getting the tensor
    tens = model_input[0]

    # checking device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    tens.to(device)

    # prediction
    with torch.no_grad():
        prediction = model(tens)
    return prediction[0].argmax(0).item()

def classify_wrapper(data: AudioData) -> int:
    """
    Ready to use classify wrapper.

    Args:
        data (AudioData): The audio data to classify.

    Returns:
        int: user's class.
    """

    model = load_model(MODEL_PATH)
    return classify(data, model)
