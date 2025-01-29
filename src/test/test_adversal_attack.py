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
from sklearn.preprocessing import LabelEncoder

from src.constants import CLASSES, MODEL_BASE_PATH
from src.model_definitions import KubaCNN1
from src.pipeline.audio_data import AudioData
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.tensor_transform import TensorTransform
from src.pipeline.classifier import Classifier

le = LabelEncoder()
le.fit(CLASSES)

# pylint: disable=line-too-long
TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f5733968_nohash_4.wav')))
ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX = str(Path.resolve(Path(f'{__file__}/../test_data/original_spectrogram')))
ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX = str(Path.resolve(Path(f'{__file__}/../test_data/adversarial_spectrogram')))
LIST_FILE = str(Path.resolve(Path(f'{__file__}/../test_data/found_files.txt')))

def create_adversarial_audio_pgd(model, audio_data, epsilon=0.05, step_size=0.01, max_iter=200):
    """
    Generate an adversarial audio example using PGD that aims to be imperceptible.

    Args:
        model: The trained PyTorch CNN model
        audio_data (AudioData): Original audio data
        epsilon (float): Perturbation strength (reduced to 0.05 for imperceptibility)
        step_size (float): Step size (reduced to 0.01 for finer optimization)
        max_iter (int): Increased iterations for better convergence

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

    # Convert model parameters to float32
    model = model.float()

    _, channels, height, width = spectrogram.shape

    # Enhanced classifier configuration
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(channels, height, width),
        nb_classes=len(CLASSES),
        preprocessing=(0, 1)  # Ensure proper scaling
    )

    # Create PGD attack with optimized parameters
    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=epsilon,
        eps_step=step_size,
        max_iter=max_iter,
        targeted=False,
        batch_size=1,
        norm=np.inf,  # Use L-infinity norm for better imperceptibility
    )

    # Generate adversarial example with early stopping
    x_test_adv = attack.generate(
        x=spectrogram,
        early_stopping=True,
        early_stopping_threshold=0.99  # Stop if confidence exceeds 99%
    )

    # Apply additional constraint to ensure imperceptibility
    delta = x_test_adv - spectrogram
    delta = np.clip(delta, -epsilon, epsilon)
    x_test_adv = spectrogram + delta

    # Convert back and denormalize
    adv_spectrogram = x_test_adv.squeeze(0) * 255.0
    original_spectrogram = og_spectrogram * 255.0

    return adv_spectrogram.astype(np.float32), original_spectrogram.astype(np.float32)

def predict_spectrogram(spectrogram_array, model):
    """
    Predicts the classification result for a given spectrogram using the specified model.
    """

    tensor_transformer = TensorTransform()
    spectrogram_tesnor = tensor_transformer.transform(spectrogram_array)
    classifier = Classifier(model)
    adv_result = classifier.predict(spectrogram_tesnor)
    predicted_label = le.inverse_transform(adv_result)[0]
    return predicted_label

def validate_perturbation(original_spec, adversarial_spec, threshold=0.1):
    """
    Validate that the perturbation is within acceptable bounds.

    Args:
        original_spec: Original spectrogram
        adversarial_spec: Adversarial spectrogram
        threshold: Maximum allowed relative difference

    Returns:
        bool: True if perturbation is acceptable
    """
    diff = np.abs(original_spec - adversarial_spec)
    relative_diff = np.mean(diff) / np.mean(original_spec)
    return relative_diff <= threshold

def analyze_files_from_list(
    path_list_file: str | Path,
    model=None,
    epsilon: float = 1.0,
    step_size: float = 0.001,
    max_iter: int = 400
) -> None:
    """
    Analyze multiple audio files for adversarial attacks.
    
    Args:
        path_list_file: File containing paths (one per line)
        model: Pre-loaded model (will load if None)
        epsilon: Perturbation strength
        step_size: Step size for PGD
        max_iter: Maximum iterations
    """
    # Load model if not provided
    if model is None:
        model = KubaCNN1.load_model(MODEL_BASE_PATH)
        if model is None:
            raise ValueError("Failed to load model")

    # Initialize statistics
    total_files = 0
    successful_attacks = 0
    original_predictions = []
    adversarial_predictions = []
    failed_files = []

    # Read and process files
    with open(path_list_file, 'r') as f:
        for line in f:
            file_path = Path(line.strip())

            try:
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                total_files += 1
                print(f"\nProcessing file {total_files}: {file_path}")

                # Load audio data
                audio_data_wav, sample_rate = sf.read(str(file_path))
                original_audio = AudioData(np.array(audio_data_wav), sample_rate)

                # Generate adversarial example
                adv_spectrogram, orig_spectrogram = create_adversarial_audio_pgd(
                    model,
                    original_audio,
                    epsilon=epsilon,
                    step_size=step_size,
                    max_iter=max_iter
                )

                # Get predictions
                orig_spectrogram = orig_spectrogram.transpose(1, 2, 0)
                orig_result = predict_spectrogram([orig_spectrogram], model=model)

                adv_spectrogram = adv_spectrogram.transpose(1, 2, 0)
                adv_result = predict_spectrogram([adv_spectrogram], model=model)

                # Store results
                original_predictions.append(orig_result)
                adversarial_predictions.append(adv_result)

                # Check if attack was successful (predictions differ)
                if orig_result != adv_result:
                    successful_attacks += 1
                    print(f"Attack successful: {orig_result} -> {adv_result}")
                else:
                    print(f"Attack failed: {orig_result} maintained")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                failed_files.append(str(file_path))

    # Print analysis results
    print("\n=== Adversarial Attack Analysis ===")
    print(f"Total files processed: {total_files}")
    print(f"Successful attacks: {successful_attacks}")
    print(f"Attack success rate: {(successful_attacks/total_files)*100:.2f}%")

    # Print class distribution changes
    print("\nClass distribution changes:")
    orig_classes = np.unique(original_predictions, return_counts=True)
    adv_classes = np.unique(adversarial_predictions, return_counts=True)

    print("\nOriginal predictions distribution:")
    for class_name, count in zip(orig_classes[0], orig_classes[1]):
        print(f"{class_name}: {count} ({count/total_files*100:.2f}%)")

    print("\nAdversarial predictions distribution:")
    for class_name, count in zip(adv_classes[0], adv_classes[1]):
        print(f"{class_name}: {count} ({count/total_files*100:.2f}%)")

    if failed_files:
        print("\nFailed to process files:")
        for file in failed_files:
            print(f"- {file}")


def main():
    """
    Enhanced main function with validation and multiple attack attempts
    """
    # Load model
    model = KubaCNN1.load_model(MODEL_BASE_PATH)
    if model is None:
        print("Failed to load model. Please check model path and file.")
        return

    # Path to your file containing the list of paths
    path_list = LIST_FILE

    epsilon_value = 1.0
    step_size = 0.001
    max_iter = 400

    try:
        analyze_files_from_list(
            path_list_file=path_list,
            epsilon=epsilon_value,
            step_size=step_size,
            max_iter=max_iter,
            model = model
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")