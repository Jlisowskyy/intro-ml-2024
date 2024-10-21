"""
Author: Jakub Pietrzak, 2024

This module provides functionality for generating and saving mel-frequency spectrograms 
from audio data using the librosa and matplotlib libraries.
"""

from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image


def gen_spectrogram(audio_data: np.array, sample_rate: int,
                    show_axis: bool = False, width: int = 400, height: int = 300) -> np.array:
    """
    Generates a mel-frequency spectrogram from audio data.

    Args:
        audio_data (np.array): Input audio signal as a NumPy array.
        sample_rate (int): Sample rate of the audio signal.
        show_axis (bool, optional): If True, display axes on the plot. Defaults to False.
        width (int, optional): Width of the output image in pixels. Defaults to 400.
        height (int, optional): Height of the output image in pixels. Defaults to 300.

    Returns:
        np.array: NumPy array representing the spectrogram image.
    """
    dpi = 100
    fmax = 8000
    s = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate,
                                       n_fft=4096, hop_length=512, n_mels=512, fmax=fmax)
    s_db = librosa.power_to_db(s, ref=np.max)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    if show_axis:
        img = librosa.display.specshow(s_db, sr=sample_rate, fmax=fmax,
                                       x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel-Frequency Spectrogram')
    else:
        img = librosa.display.specshow(s_db, sr=sample_rate, fmax=fmax, ax=ax)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    image = Image.open(buf).convert('RGB')
    image_array = np.array(image)

    buf.close()
    plt.close(fig)

    return image_array


def save_spectrogram(spectrogram: np.ndarray, file_path: str):
    """
    Saves a spectrogram image to a file.

    Args:
        spectrogram (np.ndarray): Spectrogram to save as an image.
        file_path (str): Path to save the spectrogram image file.
    """
    plt.imshow(spectrogram)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
