"""
Author: Jakub Lisowski, 2024

Test the WAV file loading and display spectrogram for the first few chunks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# pylint: disable=import-error
from src.audio.wav import load_wav_with_window


def display_spectrogram(file_path: str, num_chunks: int = 5, window_seconds: float = 0.4,
                        channel_index: int = 0):
    """
    Load a WAV file and display spectrogram for the first few chunks.

    :param file_path: Path to the WAV file
    :param num_chunks: Number of chunks to display spectrogram for
    :param window_seconds: Length of each window in seconds
    :param channel_index: Index of the audio channel to process
    """

    iterator = load_wav_with_window(file_path, window_seconds, channel_index)

    for i, chunk in enumerate(iterator):
        if i >= num_chunks:
            break

        freqs, times, sxx = spectrogram(chunk, fs=iterator.get_frame_rate(), nperseg=256)

        plt.figure(figsize=(10, 8))

        plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title(f'Spectrogram for Chunk {i + 1}')

        plt.tight_layout()
        plt.show()


def example_test_run():
    """
    Run all the tests
    """
    display_spectrogram(str(Path.resolve(Path(f'{__file__}/../test.wav'))),
                        num_chunks=10,
                        window_seconds=0.2,
                        channel_index=0)
