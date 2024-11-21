"""
Autor: Jakub Pietrzak, 2024

Modul for generating histograms from audio files
"""

import os
from pathlib import Path

import soundfile as sf
from sklearn.pipeline import Pipeline

from src.audio.audio_data import AudioData
from src.constants import HELPER_SCRIPTS_SPECTROGRAM_FOLDER_SUFFIX, \
    HELPER_SCRIPTS_HISTOGRAM_DEFAULT_DIR
from src.helper_scripts.generate_rgb_histogram import generate_rgb_histogram
from src.pipelines.audio_cleaner import AudioCleaner
from src.pipelines.audio_normalizer import AudioNormalizer
from src.pipelines.spectrogram_generator import SpectrogramGenerator


# pylint: disable=line-too-long
def create_spectrogram(directory: str, denoise: bool = False) -> str:
    """
    Function that creates spectrograms from audio files
    Args:
        denoise:  (bool): If True, denoise the audio files before creating spectrograms
        directory (str): Path to the directory with audio files.
    """
    output_directory = os.path.join(directory, HELPER_SCRIPTS_SPECTROGRAM_FOLDER_SUFFIX)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(directory)
    for index, file in enumerate(files):
        if file.endswith(".wav"):
            data, samplerate = sf.read(os.path.join(directory, file))
            audio_data = AudioData(data, samplerate)
            if denoise:
                transformation_pipeline = Pipeline(steps=[
                    ('AudioNormalizer', AudioNormalizer()),
                    ('AudioCleaner', AudioCleaner())
                ])
                transformation_pipeline.fit([audio_data])
                audio_data = transformation_pipeline.transform([audio_data])[0]
            spectrogram = SpectrogramGenerator.gen_spectrogram(audio_data, mel=True)
            SpectrogramGenerator.save_spectrogram(spectrogram, os.path.join(output_directory, file[:-4] + ".png"))
            print(f"Spectrogram: Done with {file}, {index}")
    print("Done with creating spectrograms")
    return output_directory


def process_directory(directory: str) -> None:
    """
    Main function that processes the audio files, generates spectrograms, and optionally
    creates rgb histograms.

    :param directory: Path to the directory with audio files
    """
    spectrogram_dir = create_spectrogram(directory, False)
    for file in os.listdir(spectrogram_dir):
        spectrogram_path = os.path.join(spectrogram_dir, file)
        generate_rgb_histogram(spectrogram_path)


DEFAULT_DIR = str(Path.resolve(Path(f'{__file__}/../{HELPER_SCRIPTS_HISTOGRAM_DEFAULT_DIR}')))


def main(args: list[str]) -> None:
    """
    Program entry point

    :param args: list of arguments
    """

    if len(args) == 0:
        directory = DEFAULT_DIR
    elif len(args) == 1:
        directory = args[0]
    else:
        raise ValueError("Invalid number of arguments")
    process_directory(directory)
