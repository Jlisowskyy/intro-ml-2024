"""
This module provides functionality for generating adversarial audio examples
to test the robustness of a trained PyTorch CNN model against adversarial attacks.
The adversarial examples are crafted using the Fast Gradient Sign Method (FGSM)
from the Adversarial Robustness Toolbox (ART). The module includes methods to:
"""

from pathlib import Path
import numpy as np
import torch
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import soundfile as sf
import copy

from src.constants import CLASSES, MODEL_BASE_PATH
from src.model_definitions import KubaCNN1
from src.pipeline.audio_data import AudioData
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.tensor_transform import TensorTransform
from src.pipeline.classifier import Classifier


# pylint: disable=line-too-long
TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f5733968_nohash_4.wav')))
ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX = str(Path.resolve(Path(f'{__file__}/../test_data/original_spectrogram')))
ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX = str(Path.resolve(Path(f'{__file__}/../test_data/adversarial_spectrogram')))

def create_adversarial_audio_pgd(model, audio_data, epsilon=0.1, step_size=0.02, max_iter=20):
    """
    Generate an adversarial audio example using PGD that fools the model.
    Args:
        model: The trained PyTorch CNN model
        audio_data (AudioData): Original audio data
        epsilon (float): Perturbation strength (maximum norm of the perturbation)
        step_size (float): Step size for each iteration of PGD
        max_iter (int): Number of iterations for PGD
    Returns:
        tuple: (adversarial_spectrogram, original_spectrogram) as float32 arrays
    """
    # Set model to eval mode
    model.eval()

    # Enable gradients for model parameters
    for param in model.parameters():
        param.requires_grad = True

    # Preprocess audio to create spectrogram
    og_spectrogram = process_audio(audio_data=audio_data)
    og_spectrogram = og_spectrogram.transpose(2, 0, 1)

    # Normalize to [0, 1] range and ensure float32 type
    og_spectrogram = og_spectrogram.astype(np.float32) / 255.0
    spectrogram = np.expand_dims(og_spectrogram, axis=0)

    # Convert model parameters to float32 if needed
    model = model.float()

    # Get actual dimensions
    _, channels, height, width = spectrogram.shape

    # Wrap PyTorch model with ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(channels, height, width),
        nb_classes=len(CLASSES)
    )

    # Create Projected Gradient Descent (PGD) attack
    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=epsilon,
        eps_step=step_size,
        max_iter=max_iter,
        targeted=False,
        batch_size=1
    )

    # Generate adversarial example
    x_test_adv = attack.generate(x=spectrogram)

    # Convert adversarial tensor back to numpy and denormalize
    adv_spectrogram = x_test_adv.squeeze(0) * 255.0

    # Denormalize original spectrogram
    original_spectrogram = og_spectrogram * 255.0

    # Ensure outputs are in float32
    return adv_spectrogram.astype(np.float32), original_spectrogram.astype(np.float32)

def predict_spectrogram(spectrogram_array, model):
    """
    Predicts the classification result for a given spectrogram using the specified model.
    """

    tensor_transformer = TensorTransform()
    spectrogram_tesnor = tensor_transformer.transform(spectrogram_array)
    classifier = Classifier(model)
    adv_result = classifier.predict(spectrogram_tesnor)
    return adv_result

def main():
    """
    Load a pre-trained model and audio data, and generate an adversarial audio example.
    """

    # Load your pre-trained model
    model = KubaCNN1.load_model(MODEL_BASE_PATH)
    if model is None:
        print("Failed to load model. Please check model path and file.")
        return

    # Load your original audio data
    audio_data_wav, sample_rate = sf.read(TEST_FILE_PATH)
    original_audio = AudioData(np.array(audio_data_wav), sample_rate)
    copied_audio = copy.deepcopy(original_audio)

    # original spectrogram classification
    result = model.classify([copied_audio])
    print("Orginal spectrogram prediction: " + CLASSES[result[0]])

    # Generate adversarial example
    adv_spectrogram, orig_spectrogram = create_adversarial_audio_pgd(model, original_audio)

    SpectrogramGenerator.save_spectrogram(adv_spectrogram[0], ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX + "_0.png")
    SpectrogramGenerator.save_spectrogram(adv_spectrogram[1], ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX + "_1.png")
    SpectrogramGenerator.save_spectrogram(adv_spectrogram[2], ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX + "_2.png")

    SpectrogramGenerator.save_spectrogram(orig_spectrogram[0], ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX + "_0.png")
    SpectrogramGenerator.save_spectrogram(orig_spectrogram[1], ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX + "_1.png")
    SpectrogramGenerator.save_spectrogram(orig_spectrogram[2], ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX + "_2.png")

    # transpoisng to original form
    adv_spectrogram = adv_spectrogram.transpose(1,2,0)
    adv_result = predict_spectrogram([adv_spectrogram], model=model)
    print("Adversal spectrogram prediction: " + CLASSES[adv_result[0]])
