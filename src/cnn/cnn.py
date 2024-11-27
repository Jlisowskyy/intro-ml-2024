"""
Author: Tomasz Mycielski, 2024

Implementation of the CNN
"""

from abc import ABC

import torch
from sklearn.pipeline import Pipeline
from torch import nn

from src.constants import CLASSES
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

    def classify(self, audio_data: list[AudioData],
                 pipeline: Pipeline | None = None) -> list[int]:
        """
        Classify audio data using the provided CNN model.

        Args:
            data (list[AudioData]): The audio data to classify.

        Returns:
            int: user's class.
        """

        if pipeline is None:
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

        cnn = cls(class_count=len(CLASSES))
        cnn.load_state_dict(torch.load(model_file_path, map_location=torch.device(device),
                                       weights_only=True))
        cnn.to(device)
        cnn.eval()
        return cnn
