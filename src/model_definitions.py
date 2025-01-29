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

class BasicCNN(BaseCNN):
    """
    Simplified CNN with two layers
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 128)
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

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.pool(tnnf.relu(self.conv1(dummy_input)))
            x = self.pool(tnnf.relu(self.conv2(x)))
            return x.view(-1).shape[0]


class KubaCNN1(BaseCNN):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.3)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout_fc1 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data processing method with dropout
        """
        x = self.pool1(tnnf.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool2(tnnf.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = self.pool3(tnnf.relu(self.conv3(x)))
        x = self.dropout3(x)

        x = torch.flatten(x, 1)

        x = tnnf.relu(self.fc1(x))
        x = self.dropout_fc1(x)

        x = tnnf.relu(self.fc2(x))
        x = self.dropout_fc2(x)

        x = self.fc3(x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.pool1(tnnf.relu(self.conv1(dummy_input)))
            x = self.dropout1(x)

            x = self.pool2(tnnf.relu(self.conv2(x)))
            x = self.dropout2(x)

            x = self.pool3(tnnf.relu(self.conv3(x)))
            x = self.dropout3(x)

            x = torch.flatten(x, 1)

            return x.view(-1).shape[0]

class KubaCNN2(BaseCNN):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)

        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout_fc1 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(512, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data processing method with dropout
        """
        x = self.pool1(tnnf.relu(self.conv2(tnnf.relu(self.conv1(x)))))
        x = self.dropout1(x)

        x = self.pool2(tnnf.relu(self.conv4(tnnf.relu(self.conv3(x)))))
        x = self.dropout2(x)

        x = torch.flatten(x, 1)

        x = tnnf.relu(self.fc1(x))
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        return x

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
            x = self.pool1(tnnf.relu(self.conv2(tnnf.relu(self.conv1(dummy_input)))))
            x = self.dropout1(x)

            x = self.pool2(tnnf.relu(self.conv4(tnnf.relu(self.conv3(x)))))
            x = self.dropout2(x)

            x = torch.flatten(x, 1)

            return x.view(-1).shape[0]


class ResidualCNN(BaseCNN):
    """
    Simple CNN with residual blocks
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.conv_downsample = nn.Conv2d(64, 128, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(tnnf.relu(self.bn1(self.conv1(x))))

        # First residual block
        identity = x
        x = self.res_block1(x)
        x += identity
        x = tnnf.relu(x)

        # Second residual block with dimension increase
        identity = self.conv_downsample(x)
        x = self.res_block2(x)
        x += identity
        x = tnnf.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# List of model definitions
model_definitions = [
    ModelDefinition("KubaCNN1", KubaCNN1),
    ModelDefinition("KubaCNN2", KubaCNN2),
]

BEST_MODEL = ModelDefinition("KubaCNN1", KubaCNN1)

def load_model_from_known_definitions(model_path: str) -> None | BaseCNN:
    cnn = None
    for model_definition in model_definitions:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            temp_cnn = model_definition.model()
            temp_cnn.load_model(model_path)
            cnn = temp_cnn.to(device)
            break
            # pylint: disable=broad-except, unused-variable
        except Exception as e:
            continue

    if cnn is None:
        print('No model was loaded')
        return None
    return cnn
