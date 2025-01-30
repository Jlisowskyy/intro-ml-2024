"""
Author: Michał Kwiatkowski
"""

from pathlib import Path
from itertools import islice
import soundfile as sf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.model_definitions import KubaCNN1, BaseCNN
from src.constants import MODEL_BASE_PATH, DATABASE_PATH, CLASSES
from src.pipeline.audio_data import AudioData

INPUT_DIRECTORY = f'{DATABASE_PATH}/train/audio/down'
OUTPUT_FILE = str(Path.resolve(Path(f'{__file__}/../scripts_tmp/found_files.txt')))

le = LabelEncoder()
le.fit(CLASSES)

def process_wav_files(
    root_dir: str | Path,
    output_file: str | Path,
    model: BaseCNN,
    correct_class: str,
    max_files: int = 100
) -> None:
    """
    Process up to max_files .wav files in a directory (no subdirectories).
    Append successful results to the output file.
    
    Args:
        root_dir: Directory containing wav files
        output_file: Path to the output file where successful paths will be written
        max_files: Maximum number of files to process (default: 100)
    """
    # Convert to Path object if string is provided
    root_path = Path(root_dir)
    output_path = Path(output_file)

    # Ensure the root directory exists
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get only .wav files from the root directory (no subdirectories)
    wav_files = list(islice((f for f in root_path.glob('*.wav')), max_files))

    # Process files and append results
    with open(output_path, 'a') as f:
        for wav_path in wav_files:
            try:
                # Process the file
                audio_data_wav, sample_rate = sf.read(wav_path)
                audio_data = AudioData(np.array(audio_data_wav), sample_rate)
                result = model.classify([audio_data])
                predicted_label = le.inverse_transform(result)[0]

                if predicted_label == correct_class:
                    # Append the path to output file if processing was successful
                    f.write(f"{wav_path}\n")
                    print(f"Successfully processed: {wav_path}")
            except Exception as e:
                print(f"Error processing {wav_path}: {str(e)}")


def main() -> None:
    """
    Script entry point
    """

    cnn = KubaCNN1.load_model(MODEL_BASE_PATH)
    if cnn is None:
        print('Model not found')
        return

    process_wav_files(INPUT_DIRECTORY, OUTPUT_FILE, cnn, 'down', 100)
