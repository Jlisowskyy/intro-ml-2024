from torch import nn


class ModelDefinition:
    def __init__(self, model_name: str, model: nn.Module) -> None:
        self.model_name = model_name
        self.model = model

    def __str__(self) -> str:
        return self.model_name
