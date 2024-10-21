"""
Author: Jakub Pietrzak, 2024

This module contains the function for loading a pre-trained CNN model, 
specifically the BasicCNN, and preparing it for evaluation.
"""

import torch
from ..cnn.cnn import BasicCNN

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
    cnn = BasicCNN()
    cnn.load_state_dict(torch.load(model_file_path))
    cnn.eval()  # Set the model to evaluation mode
    return cnn
