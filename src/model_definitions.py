import torch
import torch.nn as nn
import torch.functional as F

from src.cnn.cnn import BasicCNN
from src.cnn.model_definition import ModelDefinition
from src.constants import CLASSES, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH

input_shape = (3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)  # (channels, height, width)


def get_flattened_size(model, input_shape):
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
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    flattened_size = get_flattened_size(conv_layers, input_shape)
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
    flattened_size = get_flattened_size(conv_layers, input_shape)
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
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    flattened_size = get_flattened_size(conv_layers, input_shape)
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
    flattened_size = get_flattened_size(conv_layers, input_shape)
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
    flattened_size = get_flattened_size(conv_layers, input_shape)
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
