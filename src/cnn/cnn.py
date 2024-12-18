"""
Author: Tomasz Mycielski, 2024

Implementation of the CNN
"""

from abc import ABC

import torch
import torch.nn.functional as tnnf
from sklearn.pipeline import Pipeline
from torch import nn

from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.pipeline.classifier import Classifier
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.tensor_transform import TensorTransform


class BaseCNN(nn.Module, ABC):
    """
    Base class defining the CNN model functionality.
    """

    def classify(self, audio_data: list[AudioData]) -> list[int]:
        """
        Classify audio data using the provided CNN model.

        Args:
            data (list[AudioData]): The audio data to classify.

        Returns:
            int: user's class.
        """

        pipeline = Pipeline(steps=[
            ('AudioCleaner', AudioCleaner()),
            ('AudioNormalizer', AudioNormalizer()),
            ('SpectrogramGenerator', SpectrogramGenerator()),
            ('TensorTransform', TensorTransform()),
            ('Classifier', Classifier(self))
        ])

        return pipeline.predict(audio_data)

    @classmethod
    def load_model(cls, model_file_path: str) -> 'BaseCNN':
        """
        Load a pre-trained CNN model from the specified file path.

        This function initializes an instance of the calling class and loads the
        trained parameters
        Args:
            model_file_path (str): The file path to the saved model weights (state_dict).

        Returns:
            BaseCNN: An instance of the calling class with loaded weights,
            ready for inference.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        cnn = cls()
        cnn.load_state_dict(torch.load(model_file_path, map_location=torch.device(device),
                                       weights_only=True))
        cnn.eval()
        return cnn


class BasicCNN(BaseCNN):
    """
    Simplified CNN with two layers
    """

    def __init__(self, class_count=2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(111744, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data processing method
        """
        x = self.pool(tnnf.relu(self.conv1(x)))
        x = self.pool(tnnf.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = tnnf.relu(self.fc1(x))
        x = tnnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x
