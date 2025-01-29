"""
This module provides functionality for generating adversarial audio examples
to test the robustness of a trained PyTorch CNN model against adversarial attacks.
The adversarial examples are crafted using the Fast Gradient Sign Method (FGSM)
from the Adversarial Robustness Toolbox (ART). The module includes methods to:
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from src.constants import CLASSES, MODEL_BASE_PATH
from src.model_definitions import KubaCNN1, BaseCNN
from src.pipeline.audio_data import AudioData
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.tensor_transform import TensorTransform
from src.pipeline.classifier import Classifier
from src.test.test_adversal_attack import create_adversarial_audio_pgd


# pylint: disable=line-too-long
INPUT_FILE = str(Path.resolve(Path(f'{__file__}/../test_data/found_files.txt')))

def prepare_adversal_attacks(
    path_list_file: str | Path,
    model: BaseCNN
) -> None:
    """
    Read file paths from a text file and process each file.
    
    Args:
        path_list_file: File containing paths (one per line)
        processing_method: Function that takes a Path object and processes the file
    """
    # Convert to Path object if string is provided
    input_file = Path(path_list_file)

    # Check if input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read and process files
    with open(input_file, 'r') as f:
        for line in f:
            # Strip whitespace and get file path
            file_path = Path(line.strip())

            try:
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                # Process the file
                audio_data_wav, sample_rate = sf.read(file_path)
                original_audio = AudioData(np.array(audio_data_wav), sample_rate)
                adv_spectrogram, orig_spectrogram = create_adversarial_audio_pgd(model, original_audio)
                print(f"Successfully processed: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

def main():
    """
    Load a pre-trained model and audio data, and generate an adversarial audio example.
    """

    # Load your pre-trained model
    model = KubaCNN1.load_model(MODEL_BASE_PATH)
    if model is None:
        print("Failed to load model. Please check model path and file.")
        return

    prepare_adversal_attacks(INPUT_FILE, model)
