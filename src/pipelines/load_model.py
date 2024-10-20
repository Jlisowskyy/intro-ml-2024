"""
Author: Jakub Pietrzak, 2024

This module contains the function for loading a pre-trained CNN model, 
specifically the TutorialCNN, and preparing it for evaluation.
"""

import torch
from ..cnn.cnn import TutorialCNN

def load_model(model_file_path: str) -> TutorialCNN:
    """
    Load a pre-trained TutorialCNN model from the specified file path.

    This function initializes an instance of the TutorialCNN model, loads the
    trained parameters from the provided file path, and sets the model to evaluation mode.

    Args:
        model_file_path (str): The file path to the saved model weights (state_dict).

    Returns:
        TutorialCNN: An instance of the TutorialCNN model with loaded weights, 
        ready for inference.
    """
    cnn = TutorialCNN()
    cnn.load_state_dict(torch.load(model_file_path))
    cnn.eval()  # Set the model to evaluation mode
    return cnn
