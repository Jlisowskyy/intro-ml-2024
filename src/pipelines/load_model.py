"""
Author: Jakub Pietrzak, 2024

This module contains the function for loading a pre-trained CNN model, 
specifically the BasicCNN, and preparing it for evaluation.
"""
from collections.abc import Callable

import torch

from src.audio.audio_data import AudioData
from src.cnn.cnn import BasicCNN
from src.pipelines.classify import classify


def load_model(model_file_path: str) -> BasicCNN:
    """
    Load a pre-trained BasicCNN model from the specified file path.

    This function initializes an instance of the BasicCNN model, loads the
    trained parameters from the provided file path, and sets the model to evaluation mode.

    Args:
        model_file_path (str): The file path to the saved model weights (state_dict).

    Returns:
        BasicCNN: An instance of the BasicCNN model with loaded weights,
        ready for inference.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cnn = BasicCNN()
    cnn.load_state_dict(torch.load(model_file_path, map_location=torch.device(device),
                                   weights_only=True))
    cnn.eval()  # Set the model to evaluation mode
    return cnn


def get_classifier(model_path: str) -> Callable[[list[AudioData]], list[int]]:
    """
    Function loads the classifiers and return simple classification function

    :param model_path: The path to the model to validate
    """

    model = load_model(model_path)
    return lambda x: classify(x, model)
