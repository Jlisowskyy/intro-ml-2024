"""
Author: Jakub Lisowski

File tests quality of silence removal
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

from src.constants import WavIteratorType
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.wav import load_wav

TEST_FILE_NAME1 = "f5733968_nohash_4.wav"
TEST_FILE_NAME2 = "f6581345_nohash_0.wav"
TEST_FILE_NAME3 = "f2_script1_ipad_office1_35000.wav"

TEST_FILE_PATH1 = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME1}')))
TEST_FILE_OUTPUT_AFTER_PATH1 = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME1}_after_silence_removal.wav')))
TEST_FILE_PATH2 = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME2}')))
TEST_FILE_OUTPUT_AFTER_PATH2 = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME2}_after_silence_removal.wav')))
TEST_FILE_PATH3 = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME3}')))
TEST_FILE_OUTPUT_AFTER_PATH3 = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME3}_after_silence_removal.wav')))

TEST_FILES = [
    (TEST_FILE_PATH1, TEST_FILE_OUTPUT_AFTER_PATH1),
    (TEST_FILE_PATH2, TEST_FILE_OUTPUT_AFTER_PATH2),
    (TEST_FILE_PATH3, TEST_FILE_OUTPUT_AFTER_PATH3)
]


def silence_removal_test() -> None:
    """
    Run the manual test for the silence removal module.
    Displays spectrograms before and after silence removal and saves processed files.
    """

    def transform_func(x: AudioData) -> np.ndarray:
        cleaner = AudioCleaner()
        denoised = cleaner.denoise_raw(x.audio_signal, x.sample_rate)
        return AudioCleaner.remove_silence_raw(denoised, x.sample_rate)

    for test_file, test_out in TEST_FILES:
        it_cleaned = load_wav(test_file, 0, WavIteratorType.PLAIN)
        it_cleaned.transform(transform_func)
        it_cleaned.set_window_size(it_cleaned.get_num_frames())

        it = load_wav(test_file, 0, WavIteratorType.PLAIN)
        it.set_window_size(it.get_num_frames())

        original_audio = next(iter(it))
        processed_audio = next(iter(it_cleaned))

        original_audio_data = AudioData(original_audio, int(it.get_frame_rate()))
        processed_audio_data = AudioData(processed_audio, int(it.get_frame_rate()))

        original_spectrogram = SpectrogramGenerator.gen_spectrogram(original_audio_data, mel=True)
        processed_spectrogram = SpectrogramGenerator.gen_spectrogram(processed_audio_data, mel=True)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_spectrogram, aspect="auto", origin="lower")
        plt.title("Original Spectrogram")
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(1, 2, 2)
        plt.imshow(processed_spectrogram, aspect="auto", origin="lower")
        plt.title("Silence Removed Spectrogram")
        plt.colorbar(format='%+2.0f dB')

        plt.suptitle(f"Silence Removal Test - {Path(test_file).name}")
        plt.tight_layout()
        plt.show()

        write(test_out, int(it.get_frame_rate()), processed_audio)
