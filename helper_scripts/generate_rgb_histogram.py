"""
Author: Jakub Pietrzak, 2024

Modul for generating rgb histgram of spectrogram 
"""
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main(spectrogram_path: str):
    """
    Main function that processes the audio file, generates a spectrogram, and optionally 
    cleans the data.

    Args:
        sound_path (str): Path to the audio file.
        output_path (str): Optional output path for the spectrogram image.
        show (bool): Flag to show the spectrogram using matplotlib.
        mel (bool): Flag to generate a mel-frequency spectrogram.
        clean_data (bool): Flag to clean and normalize the audio data.
    """
    image=Image.open(spectrogram_path)
    image_array=np.array(image)

    r_channel = image_array[:, :, 0].flatten()
    g_channel = image_array[:, :, 1].flatten()
    b_channel = image_array[:, :, 2].flatten()


    plt.figure(figsize=(10, 5))
    plt.hist(r_channel, bins=256, color='red', alpha=0.5, label='Red Channel')
    plt.hist(g_channel, bins=256, color='green', alpha=0.5, label='Green Channel')
    plt.hist(b_channel, bins=256, color='blue', alpha=0.5, label='Blue Channel')

    plt.title('RGB Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a histogram from spectrogram.")
    parser.add_argument("spectrogram_path", type=str, help="Path to the spectrogram file.")
    
    args = parser.parse_args()

    main(args.spectrogram_path)
