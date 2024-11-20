"""
Author: Jakub Pietrzak, 2024

Modul for generating rgb histgram of spectrogram 
"""

<<<<<<< HEAD
import os
import sys
import argparse
import shutil
=======
import matplotlib.pyplot as plt
>>>>>>> origin/dev
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image

<<<<<<< HEAD
from src.audio.audio_data import AudioData
from src.audio.spectrogram import gen_spectrogram, save_spectrogram
from src.helper_scripts.spectrogram_from_npy import get_random_file_path


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))+"/temp_data"
=======
from src.constants import HELPER_SCRIPTS_HISTOGRAM_ALPHA, HELPER_SCRIPTS_HISTOGRAM_N_BINS

>>>>>>> origin/dev

def generate_rgb_histogram(spectrogram_path: str) -> None:
    """
    Generate and display the RGB histogram of a spectrogram image.

    This function reads an image from the given path, extracts the RGB channels,
    and plots a histogram for each color channel (Red, Green, and Blue). The 
    histograms represent the distribution of pixel intensities in each channel.

    Args:
        spectrogram_path (str): Path to the spectrogram image file (e.g., .png, .jpg).

    Returns:
        None: This function does not return any value. It only displays the histogram.
    """
    print("Generating RGB histogram of the spectrogram: ", spectrogram_path)
    image = Image.open(spectrogram_path)
    image_array = np.array(image)

    r_channel = image_array[:, :, 0].flatten()
    g_channel = image_array[:, :, 1].flatten()
    b_channel = image_array[:, :, 2].flatten()

    plt.figure(figsize=(10, 5))
    plt.hist(r_channel, bins=HELPER_SCRIPTS_HISTOGRAM_N_BINS, color='red',
             alpha=HELPER_SCRIPTS_HISTOGRAM_ALPHA, label='Red Channel')
    plt.hist(g_channel, bins=HELPER_SCRIPTS_HISTOGRAM_N_BINS, color='green',
             alpha=HELPER_SCRIPTS_HISTOGRAM_ALPHA, label='Green Channel')
    plt.hist(b_channel, bins=HELPER_SCRIPTS_HISTOGRAM_N_BINS, color='blue',
             alpha=HELPER_SCRIPTS_HISTOGRAM_ALPHA, label='Blue Channel')

    plt.title('RGB Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



def create_spectrogram_from_audio_file(audio_file: str) -> str:
    """
    Creates a spectrogram from an audio file and saves it as a PNG image.

    Args:
        audio_file (str): Path to the input audio file.

    Returns:
        str: Path to the saved spectrogram image.
    """
    data, samplerate = sf.read(audio_file)
    audio_data = AudioData(data, samplerate)
    spectrogram = gen_spectrogram(audio_data, mel=True)
    output_filename = f"{os.path.splitext(os.path.basename(audio_file))[0]}.png"
    output_path = os.path.join(WORKING_DIR, output_filename)
    save_spectrogram(spectrogram, output_path)
    return output_path


def process_path_and_generate_rgb_histogram(path: str, number: int = 1) -> None:
    """
    Processes the given path to generate and display RGB histograms of spectrograms.
    
    If `path` is a directory, generates histograms for `number` random audio files.
    If `path` is a spectrogram image, generates a histogram directly.
    If `path` is an audio file, generates a histogram from its spectrogram.

    Args:
        path (str): Path to a directory, spectrogram file (.jpg/.png), or audio file (.wav).
        number (int): Number of spectrograms to generate from audio files (default is 1).

    Returns:
        None: Displays RGB histograms of spectrograms directly.
    """
    if not os.path.exists(path):
        raise ValueError(f"The specified path does not exist: {path}")

    if os.path.isdir(path):
        for _ in range(number):
            file_path = get_random_file_path(path, ".wav")
            spectrogram_path=create_spectrogram_from_audio_file(file_path)
            generate_rgb_histogram(spectrogram_path)

    elif path.endswith(('.jpg', '.png')):
        generate_rgb_histogram(path)
    elif path.endswith('.wav'):
        spectrogram_path = create_spectrogram_from_audio_file(path)
        generate_rgb_histogram(spectrogram_path)
    else:
        raise ValueError(f"Unsupported file type for path: {path}")



def main(args: list[str]) -> None:
    """
    Script entry point.

    Args:
        args (list[str]): Command-line arguments.

    The script processes a given path which could be:
    - A directory containing audio files.
    - A spectrogram file (e.g., .jpg, .png).
    - An audio file (e.g., .wav).

    Options:
        -n, --number: Number of samples to generate (applicable for directories).
    """
    parser = argparse.ArgumentParser(description="Generate RGB histogram of the spectrogram")
    parser.add_argument("path", type=str,
                        help="Path to a directory, spectrogram file, or audio file")
    parser.add_argument("-n", "--number",
                        type=int, help="Number of samples to generate (for directories)")

    parsed_args = parser.parse_args(args)

    path = parsed_args.path
    number = parsed_args.number or 1

    os.makedirs(WORKING_DIR, exist_ok=True)
    process_path_and_generate_rgb_histogram(path, number)
    shutil.rmtree(WORKING_DIR)

    spectrogram_path = args[0]
    generate_rgb_histogram(spectrogram_path)
