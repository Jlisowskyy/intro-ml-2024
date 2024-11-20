"""
Author: Jakub Pietrzak, 2024

Modul for generating rgb histgram of spectrogram 
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.constants import HELPER_SCRIPTS_HISTOGRAM_ALPHA, HELPER_SCRIPTS_HISTOGRAM_N_BINS


def generate_rgb_histogram(spectrogram_path: str) -> None:
    """
    Generate RGB histogram of the spectrogram

    :param spectrogram_path: Path to the spectrogram file
   """

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


def main(args: list[str]) -> None:
    """
    Script entry point

    :param args: List of arguments
    """

    if len(args) != 1:
        raise ValueError("Invalid number of arguments")

    spectrogram_path = args[0]
    generate_rgb_histogram(spectrogram_path)
