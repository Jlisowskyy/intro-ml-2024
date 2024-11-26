"""
Author: Łukasz Kryczka
Contains functionality for manual inspection of signal and through its spectrogram
before and after a transformation provided by the user
"""
from typing import Callable, List

import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write

from src.constants import WavIteratorType
from src.pipeline.audio_data import AudioData
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.wav import load_wav
from src.test.test_file import TestFile


def test_transformation(transformation_func: Callable[[np.ndarray, int], np.ndarray],
                        transformation_name: str,
                        test_files: List[TestFile],
                        save_spectrograms=False,
                        save_audio=False,
                        plot=True) -> None:
    """
    Run the manual test for the transformation function on the provided test files.
    Displays the original and transformed audio signals
    and saves the transformed audio signal to a WAV file.

    :param transformation_func: Transformation function to apply to the audio signal
    :param transformation_name: Name of the transformation applied to the audio signal
    :param test_files: List of TestFile objects containing file paths and names
    :param save_spectrograms: Flag indicating whether to save the spectrograms as images
    :param save_audio: Flag indicating whether to save the transformed audio signal to a WAV file
    :param plot: Flag indicating whether to plot the spectrograms
    """
    for test_file in test_files:
        it_transformed = load_wav(test_file.file_path, 0, WavIteratorType.PLAIN)
        it_transformed.transform(transformation_func)
        it_transformed.set_window_size(it_transformed.get_num_frames())

        it = load_wav(test_file.file_path, 0, WavIteratorType.PLAIN)
        it.set_window_size(it.get_num_frames())

        original_audio = next(iter(it))
        processed_audio = next(iter(it_transformed))

        original_audio_data = AudioData(original_audio, int(it.get_frame_rate()))
        processed_audio_data = AudioData(processed_audio, int(it.get_frame_rate()))

        original_spectrogram = SpectrogramGenerator.gen_spectrogram(original_audio_data, mel=True)
        processed_spectrogram = SpectrogramGenerator.gen_spectrogram(processed_audio_data, mel=True)

        if plot:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(original_spectrogram, aspect="auto", origin="lower")
            plt.title("Original Spectrogram")
            plt.colorbar(format='%+2.0f dB')

            plt.subplot(1, 2, 2)
            plt.imshow(processed_spectrogram, aspect="auto", origin="lower")
            plt.title(f"Spectrogram after {transformation_name}")
            plt.colorbar(format='%+2.0f dB')

            plt.suptitle(f"{transformation_name} Test - {test_file.file_name}")
            plt.tight_layout()
            plt.show()

        if save_spectrograms:
            SpectrogramGenerator.save_spectrogram(original_spectrogram,
                                                  test_file.file_path.replace(".wav", ".png"))
            SpectrogramGenerator.save_spectrogram(processed_spectrogram,
                                                  test_file.get_transformed_file_path_out(
                                                      transformation_name).replace(
                                                      ".wav", ".png"))

        if save_audio:
            write(test_file.get_transformed_file_path_out(transformation_name),
                  int(it.get_frame_rate()), processed_audio)
