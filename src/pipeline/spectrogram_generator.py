"""
Author: Michał Kwiatkowski, Jakub Pietrzak

This module contains the SpectrogramGenerator class, which provides functionality
for generating mel-frequency spectrograms from audio data.
"""
from io import BytesIO

import librosa
import matplotlib
import numpy as np
import torch
import torchaudio
import torchaudio.transforms
from PIL import Image
from matplotlib import pyplot as plt

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.constants import (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, DENOISE_FREQ_HIGH_CUT,
                           SPECTROGRAM_DPI, SPECTROGRAM_N_FFT,
                           SPECTROGRAM_HOP_LENGTH,
                           SPECTROGRAM_N_MELS, DENOISE_FREQ_LOW_CUT)
from src.pipeline.audio_data import AudioData


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
        Generates a spectrogram from audio data using GPU acceleration via torchaudio.
        Args:
            audio_data: AudioData object containing the audio signal and sample rate.
            mel (bool, optional): If True, generate mel spectrogram. Defaults to False.
            show_axis (bool, optional): If True, display axes on the plot. Defaults to False.
            width (int, optional): Width of the output image in pixels. Defaults to 400.
            height (int, optional): Height of the output image in pixels.
        Returns:
            np.ndarray: Image array of the spectrogram.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        waveform = torch.tensor(audio.audio_signal).float().to(device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if mel:
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=audio.sample_rate,
                n_fft=SPECTROGRAM_N_FFT,
                hop_length=SPECTROGRAM_HOP_LENGTH,
                n_mels=SPECTROGRAM_N_MELS,
                f_max=DENOISE_FREQ_HIGH_CUT,
                f_min=DENOISE_FREQ_LOW_CUT
            ).to(device)

            spec = mel_transform(waveform)
            spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
        else:
            spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=SPECTROGRAM_N_FFT,
                hop_length=SPECTROGRAM_HOP_LENGTH,
                power=2.0
            ).to(device)

            spec = spec_transform(waveform)
            spec_db = torchaudio.transforms.AmplitudeToDB()(spec)

        # Move back to CPU for plotting
        spec_db = spec_db.cpu().numpy()[0]

        dpi = SPECTROGRAM_DPI
        # Create Figure object directly instead of using pyplot
        fig = plt.Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        cmap = plt.colormaps['magma']
        if mel:
            img = librosa.display.specshow(
                spec_db,
                sr=audio.sample_rate,
                fmax=DENOISE_FREQ_HIGH_CUT,
                x_axis='time',
                y_axis='mel',
                ax=ax,
                cmap=cmap
            )
        else:
            img = librosa.display.specshow(
                spec_db,
                sr=audio.sample_rate,
                x_axis='time',
                y_axis='log',
                ax=ax,
                cmap=cmap
            )

        if show_axis:
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title('Spectrogram')
        else:
            ax.axis('off')
            fig.tight_layout(pad=0)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        with BytesIO() as buf:
            canvas.print_png(buf)
            buf.seek(0)
            image = np.array(Image.open(buf).convert('RGB'))

        # Clean up
        fig.clear()
        return image

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
