"""
Author: Michał Kwiatkowski
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch.nn as nn
import torch.optim as optim
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt

from src.model_definitions import DeeperCNN
from src.pipeline.audio_data import AudioData
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.cnn.model_definition import ModelDefinition
from src.constants import MODEL_BASE_PATH, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH
from src.cnn.model_api import classify_file


def visualize_spectrogram(spectrogram, title="Spectrogram", output_file=None):
    """
    Visualizes and optionally saves a spectrogram.

    Args:
    spectrogram (ndarray): The spectrogram to visualize.
    title (str): The title of the plot.
    output_file (str, optional): If provided, saves the plot as an image file.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram[0, 0], cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    if output_file:
        plt.savefig(output_file)  # Save to file
        print(f"Plot saved as {output_file}")
    else:
        plt.show()


def preprocess_audio(file_path, target_shape):
    """
    Preprocesses an audio file by converting it into a spectrogram and resizing it.

    Args:
    file_path (str): The path to the audio file.
    target_shape (tuple): The target shape of the spectrogram.

    Returns:
    ndarray: The preprocessed spectrogram.
    """
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Convert to a spectrogram (Mel Spectrogram)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=target_shape[0])

    # Normalize to [0, 1]
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

    # Resize to match the input shape
    spectrogram = np.resize(spectrogram, target_shape)

    # Add channel dimension
    spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, H, W)
    return spectrogram


# pylint: disable=line-too-long
TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f5733968_nohash_4.wav')))


def example_test_run():
    """
    Example test run demonstrating the attack on the audio classifier.
    """
    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=DeeperCNN().parameters(), lr=0.001)  # Dummy optimizer

    # Wrap your DeeperCNN model into a PyTorchClassifier
    classifier = PyTorchClassifier(
        model=DeeperCNN(),
        clip_values=(0.0, 1.0),
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH),
        nb_classes=11  # Adjust the number of classes as necessary
    )

    wav_file = TEST_FILE_PATH

    # Preprocess audio into a spectrogram
    spectrogram = preprocess_audio(wav_file, target_shape=(3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH))  # Ensure 3 channels

    # Initialize Fast Gradient Method
    attack = FastGradientMethod(estimator=classifier, eps=0.1)  # Adjust epsilon as needed

    # Generate adversarial example
    adversarial_example = attack.generate(x=spectrogram)

    # Visualize and save spectrograms
    visualize_spectrogram(spectrogram, title="Original Spectrogram", output_file="original_spectrogram.png")
    visualize_spectrogram(adversarial_example, title="Adversarial Spectrogram", output_file="adversarial_spectrogram.png")

    # Save or evaluate the adversarial spectrogram
    print("Adversarial example generated!")
