"""
Author: Michał Kwiatkowski

This module contains the Classifier class, which provides functionality
for fitting a model and making predictions based on audio data.
"""

import torch
from torch import nn


class Classifier:
    """
    A simple classifier for predicting labels from audio data.

    This class provides methods to fit the model using training data and
    to predict labels for new data inputs.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the Classifier instance.
        """

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    # pylint: disable=unused-argument
    def fit(self, x_data: list[torch.Tensor], y_data: list[int]) -> 'Classifier':
        """
        Fit the classifier to the training data.

        Args:
            x_data (list[torch.Tensor]): A list of input data arrays (features).
            y_data (list[int]): A list of labels corresponding to the input data.

        Returns:
            self: Returns an instance of the fitted classifier.
        """
        return self

    def predict(self, x_data: list[torch.Tensor]) -> list[int]:
        """
        Predict labels for the given input data.

        Args:
            x_data (list[torch.Tensor]): A list of input data arrays (features) 
            to predict labels for.

        Returns:
            list[int]: A list of predicted labels.
        """

        predictions = []

        with torch.no_grad():
            for tens in x_data:
                tens = tens.to(self.device)

                prediction = self.model(tens)
                predicted_label = prediction.argmax(-1).item()
                predictions.append(predicted_label)

        return predictions
