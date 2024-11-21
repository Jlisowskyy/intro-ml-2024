"""
Author: Michał Kwiatkowski, Jakub Pietrzak

This module contains the SpectrogramGenerator class, which provides functionality
for generating mel-frequency spectrograms from audio data.
"""
from io import BytesIO

import numpy as np
import librosa
from PIL import Image
from librosa import feature
from matplotlib import pyplot as plt

from src.pipeline.audio_data import AudioData
from src.constants import (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, DENOISE_FREQ_HIGH_CUT,
                           SPECTROGRAM_DPI, SPECTROGRAM_N_FFT,
                           SPECTROGRAM_HOP_LENGTH,
                           SPECTROGRAM_N_MELS)



class SpectrogramGenerator:
    """
    A class to generate mel-frequency spectrogram's from audio data.

    This class provides methods to fit the model (if applicable) and to transform
    audio data into spectrogram representations.
    """

    def __init__(self) -> None:
        """
        Initializes the SpectrogramGenerator instance.
        """
        return

    @staticmethod
    def gen_spectrogram(audio: AudioData, mel: bool = False,
                        show_axis: bool = False, width: int = SPECTROGRAM_WIDTH,
                        height: int = SPECTROGRAM_HEIGHT) -> np.ndarray:
        """
        Generates a normal spectrogram from audio data.
        Args:
            audio_data: AudioData object containing the audio signal and sample rate.
            show_axis (bool, optional): If True, display axes on the plot. Defaults to False.
            width (int, optional): Width of the output image in pixels. Defaults to 400.
            height (int, optional): Height of the output image in pixels.
        Returns:
            np.ndarray: Image array of the spectrogram.
        """
        dpi = SPECTROGRAM_DPI
        if mel:
            s = feature.melspectrogram(y=audio.audio_signal, sr=audio.sample_rate,
                                       n_fft=SPECTROGRAM_N_FFT, hop_length=SPECTROGRAM_HOP_LENGTH,
                                       n_mels=SPECTROGRAM_N_MELS,
                                       fmax=DENOISE_FREQ_HIGH_CUT)
            s_db = librosa.power_to_db(s, ref=np.max)
        else:
            s = librosa.stft(audio.audio_signal, n_fft=SPECTROGRAM_N_FFT,
                             hop_length=SPECTROGRAM_HOP_LENGTH)
            s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)

        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

        if mel:
            img = librosa.display.specshow(s_db, sr=audio.sample_rate, fmax=DENOISE_FREQ_HIGH_CUT,
                                           x_axis='time', y_axis='mel', ax=ax)
        else:
            img = librosa.display.specshow(s_db, sr=audio.sample_rate, x_axis='time',
                                           y_axis='log', ax=ax)

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

    @staticmethod
    def save_spectrogram(spectrogram: np.ndarray, file_path: str) -> None:
        """
        Saves a spectrogram image to a file.

        Args:
            spectrogram (np.ndarray): Spectrogram to save as an image.
            file_path (str): Path to save the spectrogram image file.
        """
        plt.imshow(spectrogram)
        plt.axis('off')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'SpectrogramGenerator':
        """
        Fit the generator to the audio data (if necessary).

        Args:
            x_data (list[AudioData]): A list of AudioData instances.
            y_data (list[int], optional): A list of labels (if applicable).

        Returns:
            self: Returns an instance of the fitted generator.
        """
        return self

    # pylint: disable=line-too-long
    def transform(self, audio_data_list: list[AudioData]) -> list[np.ndarray]:
        """
        Transform audio data into mel-frequency spectrogram's.

        Args:
            audio_data_list (list[AudioData]): A list of AudioData instances 
            to be transformed into spectrogram's.

        Returns:
            list[np.ndarray]: A list of NumPy arrays representing the spectrogram's.
        """
        spectrogram_data = []
        for audio_data in audio_data_list:
            spectrogram = SpectrogramGenerator.gen_spectrogram(audio_data, mel=True)
            spectrogram_data.append(spectrogram)
        return spectrogram_data
