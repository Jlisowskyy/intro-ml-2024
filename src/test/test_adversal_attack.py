# """
# Author: Michał Kwiatkowski
# """

# from pathlib import Path
# import numpy as np
# import soundfile as sf
# import librosa
# import torch.nn as nn
# import torch.optim as optim
# from art.attacks.evasion import FastGradientMethod
# from art.estimators.classification import PyTorchClassifier
# import matplotlib.pyplot as plt

# from src.model_definitions import DeeperCNN
# from src.pipeline.audio_data import AudioData
# from src.pipeline.spectrogram_generator import SpectrogramGenerator
# from src.cnn.model_definition import ModelDefinition
# from src.constants import MODEL_BASE_PATH, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH
# from src.cnn.model_api import classify_file


# def visualize_spectrogram(spectrogram, title="Spectrogram", output_file=None):
#     """
#     Visualizes and optionally saves a spectrogram.

#     Args:
#     spectrogram (ndarray): The spectrogram to visualize.
#     title (str): The title of the plot.
#     output_file (str, optional): If provided, saves the plot as an image file.
#     """
#     plt.figure(figsize=(10, 5))
#     plt.imshow(spectrogram[0, 0], cmap="viridis", origin="lower", aspect="auto")
#     plt.colorbar(format="%+2.0f dB")
#     plt.title(title)
#     plt.xlabel("Time")
#     plt.ylabel("Frequency")

#     if output_file:
#         plt.savefig(output_file)  # Save to file
#         print(f"Plot saved as {output_file}")
#     else:
#         plt.show()


# def preprocess_audio(file_path, target_shape):
#     """
#     Preprocesses an audio file by converting it into a spectrogram and resizing it.

#     Args:
#     file_path (str): The path to the audio file.
#     target_shape (tuple): The target shape of the spectrogram.

#     Returns:
#     ndarray: The preprocessed spectrogram.
#     """
#     # Load the audio file
#     audio, sr = librosa.load(file_path, sr=None)

#     # Convert to a spectrogram (Mel Spectrogram)
#     spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=target_shape[0])

#     # Normalize to [0, 1]
#     spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
#     spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

#     # Resize to match the input shape
#     spectrogram = np.resize(spectrogram, target_shape)

#     # Add channel dimension
#     spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, H, W)
#     return spectrogram


# # pylint: disable=line-too-long
# TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f5733968_nohash_4.wav')))


# def example_test_run():
#     """
#     Example test run demonstrating the attack on the audio classifier.
#     """
#     # Define the loss function and optimizer
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(params=DeeperCNN().parameters(), lr=0.001)  # Dummy optimizer

#     # Wrap your DeeperCNN model into a PyTorchClassifier
#     classifier = PyTorchClassifier(
#         model=DeeperCNN(),
#         clip_values=(0.0, 1.0),
#         loss=loss_fn,
#         optimizer=optimizer,
#         input_shape=(3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH),
#         nb_classes=11  # Adjust the number of classes as necessary
#     )

#     wav_file = TEST_FILE_PATH

#     # Preprocess audio into a spectrogram
#     spectrogram = preprocess_audio(wav_file, target_shape=(3, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH))  # Ensure 3 channels

#     # Initialize Fast Gradient Method
#     attack = FastGradientMethod(estimator=classifier, eps=0.1)  # Adjust epsilon as needed

#     # Generate adversarial example
#     adversarial_example = attack.generate(x=spectrogram)

#     # Visualize and save spectrograms
#     visualize_spectrogram(spectrogram, title="Original Spectrogram", output_file="original_spectrogram.png")
#     visualize_spectrogram(adversarial_example, title="Adversarial Spectrogram", output_file="adversarial_spectrogram.png")

#     # Save or evaluate the adversarial spectrogram
#     print("Adversarial example generated!")

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from pathlib import Path
import soundfile as sf

from src.cnn.cnn import BaseCNN
from src.constants import CLASSES, MODEL_BASE_PATH
from src.model_definitions import DeeperCNN
from src.pipeline.audio_data import AudioData
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.tensor_transform import TensorTransform

TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f5733968_nohash_4.wav')))

def create_adversarial_audio(model, audio_data, true_label, epsilon=0.01):
    """
    Generate an adversarial audio example that fools the model.
    
    Args:
        model (DeeperCNN): The trained PyTorch CNN model
        audio_data (AudioData): Original audio data
        true_label (int): The correct classification label
        epsilon (float): Perturbation strength
    
    Returns:
        AudioData: Adversarial audio example
    """
    # Convert audio to PyTorch tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spectrogram = process_audio(audio_data=audio_data)
    SpectrogramGenerator.save_spectrogram(spectrogram=spectrogram, file_path="original_spectrogram.png")

    spectrogram = spectrogram.transpose(2, 0, 1)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Final shape: (1, 3, 400, 300)

    print(spectrogram.shape)

    # Wrap PyTorch model with ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 255),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, spectrogram.shape[0], spectrogram.shape[1]),
        nb_classes=len(CLASSES)
    )

    # Create Fast Gradient Sign Method (FGSM) attack
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)

    # Generate adversarial example
    x_test_adv = attack.generate(x=spectrogram)
    print(x_test_adv.shape)

    # Convert adversarial tensor back to numpy for AudioData
    adv_spectrogram = x_test_adv.squeeze(0)
    adv_spectrogram_channel = adv_spectrogram[0]

    # Optional: Save adversarial spectrogram for visualization
    SpectrogramGenerator.save_spectrogram(adv_spectrogram_channel, 'adversarial_spectrogram.png')

    # Reconstruct AudioData (this is a simplification and might need audio reconstruction)
    return AudioData(adv_spectrogram_channel.astype(np.float32), audio_data.sample_rate)

def verify_attack(model, original_audio, adversarial_audio):
    """
    Verify the effectiveness of the adversarial attack
    
    Args:
        model (DeeperCNN): Trained model
        original_audio (AudioData): Original audio
        adversarial_audio (AudioData): Perturbed audio
    
    Returns:
        tuple: (original prediction, adversarial prediction)
    """
    original_prediction = model.classify([original_audio])[0]
    adv_prediction = model.classify([adversarial_audio])[0]

    print(f"Original Prediction: {CLASSES[original_prediction]}")
    print(f"Adversarial Prediction: {CLASSES[adv_prediction]}")

    return original_prediction, adv_prediction

# Example usage
def main():
    # Load your pre-trained model
    model = DeeperCNN.load_model(MODEL_BASE_PATH)

    # Load your original audio data
    audio_data_wav, sample_rate = sf.read(TEST_FILE_PATH)
    original_audio = AudioData(np.array(audio_data_wav), sample_rate)
    true_label = CLASSES[1] # no

    # Generate adversarial example
    adversarial_audio = create_adversarial_audio(model, original_audio, true_label)

    # Verify attack effectiveness
    verify_attack(model, original_audio, adversarial_audio)
