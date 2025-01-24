"""
This module contains the ModelDefinition class,
which is used to store the name of a model and the model itself.
"""
from torch import nn


class ModelDefinition:
    """
    A class to store the name of a model and the model itself.
    """
    def __init__(self, model_name: str, model: nn.Module) -> None:
        self.model_name = model_name
        self.model = model

    def __str__(self) -> str:
        return self.model_name
