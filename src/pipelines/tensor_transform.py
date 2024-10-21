"""
Author: Michał Kwiatkowski

Module for transforming spectrogram data into tensors suitable for PyTorch models.
"""

import numpy as np
import torch

class TensorTransform:
    """
    A class used to fit and transform spectrogram data into PyTorch tensors.
    """
    def __init__(self):
        """
        Initializes the TensorTransform object.
        """
        return

    # pylint: disable=unused-argument
    def fit(self, x_data: list[np.ndarray], y_data: list[int] = None):
        """
        Placeholder fit method. Does nothing as no fitting is needed.
        
        Parameters:
        x_data (list[np.ndarray]): List of input spectrogram data.
        y_data (list[int], optional): List of labels. Defaults to None.
        
        Returns:
        self: The fitted transformer instance.
        """
        return self

    def transform(self, x_data: list[np.ndarray], y_data: list[int] = None) -> list[torch.Tensor]:
        """
        Transforms a list of spectrograms into PyTorch tensors.

        Parameters:
        x_data (list[np.ndarray]): List of input spectrogram data.

        Returns:
        list[torch.Tensor]: List of transformed spectrograms as tensors.
        """
        tensors = []
        for spectrogram in x_data:
            tens = torch.from_numpy(spectrogram).type(torch.float32)
            tens = torch.rot90(tens, dims=(0, 2))  # Put channel count as the first axis
            tens = tens[None, :, :, :]  # Add batch dimension
            tensors.append(tens)
        return tensors
