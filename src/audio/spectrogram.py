"""
Author: Jakub Pietrzak, 2024

File collects various logic correlated to spectrogram processing.
"""

from io import BytesIO

import librosa
import numpy as np
from PIL import Image
from librosa import feature
from matplotlib import pyplot as plt

from src.audio.audio_data import AudioData
from src.constants import (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, DENOISE_FREQ_HIGH_CUT,
                           SPECTROGRA_DPI, SPECTROGRAM_N_FFT,
                           SPECTROGRAM_HOP_LENGTH,
                           SPECTROGRAM_N_MELS)


def gen_spectrogram(audio_data: AudioData,
                    show_axis: bool = False, width: int = SPECTROGRAM_WIDTH,
                    height: int = SPECTROGRAM_HEIGHT) -> np.ndarray:
    """
    # TODO: MISSING DOCSTRING
    Args:
        audio_data:
        show_axis:
        width:
        height:

    Returns:

    """
    dpi = SPECTROGRA_DPI
    s = librosa.stft(audio_data.audio_signal, n_fft=SPECTROGRAM_N_FFT,
                     hop_length=SPECTROGRAM_HOP_LENGTH)
    s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    img = librosa.display.specshow(s_db, sr=audio_data.sample_rate,
                                   x_axis='time', y_axis='log', ax=ax)
    if show_axis:
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Spectrogram')
    else:
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


def gen_mel_spectrogram(audio_data: AudioData,
                        show_axis: bool = False, width: int = SPECTROGRAM_WIDTH,
                        height: int = SPECTROGRAM_HEIGHT) -> np.ndarray:
    """
    # TODO: MISSING DOCSTRING
    Args:
        audio_data:
        show_axis:
        width:
        height:

    Returns:

    """
    dpi = SPECTROGRA_DPI
    s = feature.melspectrogram(y=audio_data.audio_signal, sr=audio_data.sample_rate,
                               n_fft=SPECTROGRAM_N_FFT, hop_length=SPECTROGRAM_HOP_LENGTH,
                               n_mels=SPECTROGRAM_N_MELS, fmax=DENOISE_FREQ_HIGH_CUT)
    s_db = librosa.power_to_db(s, ref=np.max)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    img = librosa.display.specshow(s_db, sr=audio_data.sample_rate, fmax=DENOISE_FREQ_HIGH_CUT,
                                   x_axis='time', y_axis='mel', ax=ax)
    if show_axis:
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel-Frequency Spectrogram')
    else:
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
