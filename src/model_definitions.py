"""
This module contains a list of model definitions that can be used to train a CNN model.
It is used by the train.py script to train multiple models and compare their performance.
"""

import torch
from torch import nn
from torch.nn import functional as tnnf

from src.cnn.cnn import BaseCNN
from src.cnn.model_definition import ModelDefinition
from src.constants import CLASSES, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH

# pylint: disable=missing-class-docstring,missing-function-docstring


class SimpleCNN(BaseCNN):
    """
    A basic CNN with two convolutional layers and two fully connected layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            return x.view(-1).shape[0]


class DeeperCNN(BaseCNN):
    """
    A deeper CNN with three convolutional layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, len(CLASSES))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.relu1(self.conv1(dummy_input))
            x = self.relu2(self.conv2(x))
            x = self.pool1(x)
            x = self.relu3(self.conv3(x))
            x = self.pool2(x)
            return x.view(-1).shape[0]


class WideCNN(BaseCNN):
    """
    A CNN with wider convolutional layers (more filters).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, len(CLASSES))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            return x.view(-1).shape[0]


class DropoutCNN(BaseCNN):
    """
    A CNN that includes dropout layers to reduce overfitting.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, len(CLASSES))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.dropout1(x)
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.dropout2(x)
            return x.view(-1).shape[0]


class BatchNormCNN(BaseCNN):
    """
    A CNN that includes batch normalization layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        x = self.pool1(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.pool1(self.relu1(self.batchnorm1(self.conv1(dummy_input))))
            x = self.pool2(self.relu2(self.batchnorm2(self.conv2(x))))
            return x.view(-1).shape[0]


class BasicCNN(BaseCNN):
    """
    Simplified CNN with two layers
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(111744, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, len(CLASSES))

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


# List of model definitions
model_definitions = [
    ModelDefinition('BasicCNN', BasicCNN()),
    ModelDefinition('SimpleCNN', SimpleCNN()),
    ModelDefinition('DeeperCNN', DeeperCNN()),
    ModelDefinition('WideCNN', WideCNN()),
    ModelDefinition('DropoutCNN', DropoutCNN()),
    ModelDefinition('BatchNormCNN', BatchNormCNN()),
]
