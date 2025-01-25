import numpy as np
from pathlib import Path
import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import soundfile as sf

from src.constants import CLASSES, MODEL_BASE_PATH
from src.model_definitions import DeeperCNN
from src.pipeline.audio_data import AudioData
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.spectrogram_generator import SpectrogramGenerator

TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f5733968_nohash_4.wav')))
ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX = str(Path.resolve(Path(f'{__file__}/../test_data/original_spectrogram')))
ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX = str(Path.resolve(Path(f'{__file__}/../test_data/adversarial_spectrogram')))

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
    og_spectrogram = process_audio(audio_data=audio_data)
    og_spectrogram = og_spectrogram.transpose(2, 0, 1)
    spectrogram = np.expand_dims(og_spectrogram, axis=0)

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

    # Convert adversarial tensor back to numpy for AudioData
    adv_spectrogram = x_test_adv.squeeze(0)

    # Optional: Save adversarial spectrogram for visualization
    SpectrogramGenerator.save_spectrogram(adv_spectrogram[0], ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX + "_0.png")
    SpectrogramGenerator.save_spectrogram(adv_spectrogram[1], ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX + "_1.png")
    SpectrogramGenerator.save_spectrogram(adv_spectrogram[2], ADVERSARIAL_SPECTROGRAM_OUTPUT_PREFIX + "_2.png")

    SpectrogramGenerator.save_spectrogram(og_spectrogram[0], ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX + "_0.png")
    SpectrogramGenerator.save_spectrogram(og_spectrogram[1], ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX + "_1.png")
    SpectrogramGenerator.save_spectrogram(og_spectrogram[2], ORIGINAL_SPECTROGRAM_OUTPUT_PREFIX + "_2.png")

    difference_indices = np.where(adv_spectrogram != og_spectrogram)

    if difference_indices[0].size == 0:
        print("The adversarial spectrogram and the original spectrogram are exactly the same.")
    else:
        print("The adversarial spectrogram and the original spectrogram are different.")
        print(f"Differences found at indices: {difference_indices}")

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

def main():
    # Load your pre-trained model
    model = DeeperCNN.load_model(MODEL_BASE_PATH)

    # Load your original audio data
    audio_data_wav, sample_rate = sf.read(TEST_FILE_PATH)
    original_audio = AudioData(np.array(audio_data_wav), sample_rate)
    true_label = CLASSES[1] # no

    # Generate adversarial example
    create_adversarial_audio(model, original_audio, true_label)