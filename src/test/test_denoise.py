"""
Author: Łukasz Kryczka, 2024

Manual test cases for denoise module using a pretrained DNS64 model.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

from src.pipeline.wav import load_wav, WavIteratorType, AudioDataIterator
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData

TEST_WINDOW_LENGTH_MS = 999999
TEST_FILE_NAME = "f2_script1_ipad_office1_35000.wav"
TEST_FILE_PATH = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME}')))
TEST_FILE_OUTPUT_BEFORE_PATH = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME}_before_denoise.wav')))
TEST_FILE_OUTPUT_AFTER_PATH = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME}_after_denoise.wav')))


def denoise_test_manual():
    """
    Run the manual test for the denoise module.
    It loads a WAV file, adds noise to the audio signal, denoises it using the
    AudioCleaner class, and displays the original and denoised spectrograms.

    The 'original + noise' and 'denoised' audio signals are saved to WAV files.
    """
    iterator = load_wav(TEST_FILE_PATH, 0, WavIteratorType.PLAIN)
    frame_rate = iterator.get_frame_rate()
    window_size_frames = int(TEST_WINDOW_LENGTH_MS * frame_rate / 1000)
    iterator.set_window_size(window_size_frames)

    iterator = AudioDataIterator(iterator)

    for i, audio_data in enumerate(iterator):
        if i > 0:
            break
        # Add some noise to the audio signal
        noise = np.random.normal(0, 0.02, audio_data.audio_signal.shape)
        audio_data.audio_signal += noise

        audio_cleaner = AudioCleaner()

        # audio_data = audio_normalizer.normalize(audio_data, NormalizationType.MEAN_VARIANCE)

        # Deep copy the audio data to avoid modifying the original signal
        denoised_audio_data = AudioData(np.copy(audio_data.audio_signal), audio_data.sample_rate)
        denoised_audio_data = audio_cleaner.denoise(denoised_audio_data)

        # Generate spectrograms for the original and denoised audio signals
        original_spectrogram = SpectrogramGenerator.gen_spectrogram(audio_data, mel=True)
        denoised_spectrogram = SpectrogramGenerator.gen_spectrogram(denoised_audio_data, mel=True)

        # Display the original and denoised spectrograms
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_spectrogram, aspect="auto", origin="lower")
        plt.title("Original Spectrogram")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(denoised_spectrogram, aspect="auto", origin="lower")
        plt.title("Denoised Spectrogram")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # Save the denoised audio signal to a WAV file
        write(TEST_FILE_OUTPUT_BEFORE_PATH,
              audio_data.sample_rate, audio_data.audio_signal)
        write(TEST_FILE_OUTPUT_AFTER_PATH,
              denoised_audio_data.sample_rate, denoised_audio_data.audio_signal)
