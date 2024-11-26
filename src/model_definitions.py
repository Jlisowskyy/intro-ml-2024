"""
This module contains a list of model definitions that can be used to train a CNN model.
It is used by the train.py script to train multiple models and compare their performance.
"""
import torch
import torch.nn as nn

from src.cnn.cnn import BasicCNN
from src.cnn.model_definition import ModelDefinition
from src.constants import CLASSES, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH

spectrogram_input_shape = (3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)  # (channels, height, width)


def get_flattened_size(model, input_shape):
    """
    Get the size of the output tensor after passing through the model.
    """
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = model(dummy_input)
        return output.view(-1).shape[0]


model_definitions = []


# Default CNN model
default_cnn = BasicCNN(len(CLASSES))
model_definitions.append(ModelDefinition('DefaultCNN', default_cnn))


# A basic CNN with two convolutional layers and two fully connected layers
def create_simple_cnn():
    """
    Create a simple CNN model with two convolutional layers and two fully connected layers.
    """
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    flattened_size = get_flattened_size(conv_layers, spectrogram_input_shape)
    fc_layers = nn.Sequential(
        nn.Linear(flattened_size, 128),
        nn.ReLU(),
        nn.Linear(128, len(CLASSES))
    )
    model = nn.Sequential(
        conv_layers,
        nn.Flatten(),
        fc_layers
    )
    return ModelDefinition('SimpleCNN', model)


model_definitions.append(create_simple_cnn())


# A deeper CNN with three convolutional layers
def create_deeper_cnn():
    """
    Create a deeper CNN model with three convolutional layers.
    """
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    flattened_size = get_flattened_size(conv_layers, spectrogram_input_shape)
    fc_layers = nn.Sequential(
        nn.Linear(flattened_size, 256),
        nn.ReLU(),
        nn.Linear(256, len(CLASSES))
    )
    model = nn.Sequential(
        conv_layers,
        nn.Flatten(),
        fc_layers
    )
    return ModelDefinition('DeeperCNN', model)


model_definitions.append(create_deeper_cnn())


# A CNN with wider convolutional layers (more filters)
def create_wide_cnn():
    """
    Create a CNN model with wider convolutional layers (more filters).
    """
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    flattened_size = get_flattened_size(conv_layers, spectrogram_input_shape)
    fc_layers = nn.Sequential(
        nn.Linear(flattened_size, 256),
        nn.ReLU(),
        nn.Linear(256, len(CLASSES))
    )
    model = nn.Sequential(
        conv_layers,
        nn.Flatten(),
        fc_layers
    )
    return ModelDefinition('WideCNN', model)


model_definitions.append(create_wide_cnn())


# A CNN that includes dropout layers to reduce overfitting
def create_dropout_cnn():
    """
    Create a CNN model with dropout layers to reduce overfitting.
    """
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Conv2d(32, 64, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25)
    )
    flattened_size = get_flattened_size(conv_layers, spectrogram_input_shape)
    fc_layers = nn.Sequential(
        nn.Linear(flattened_size, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(CLASSES))
    )
    model = nn.Sequential(
        conv_layers,
        nn.Flatten(),
        fc_layers
    )
    return ModelDefinition('DropoutCNN', model)


model_definitions.append(create_dropout_cnn())


# A CNN that includes batch normalization layers
def create_batchnorm_cnn():
    """
    Create a CNN model with batch normalization layers.
    """
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    flattened_size = get_flattened_size(conv_layers, spectrogram_input_shape)
    fc_layers = nn.Sequential(
        nn.Linear(flattened_size, 128),
        nn.ReLU(),
        nn.Linear(128, len(CLASSES))
    )
    model = nn.Sequential(
        conv_layers,
        nn.Flatten(),
        fc_layers
    )
    return ModelDefinition('BatchNormCNN', model)


model_definitions.append(create_batchnorm_cnn())
