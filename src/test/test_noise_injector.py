"""
Author: Łukasz Kryczka, 2024

Manual test cases for denoise module using a pretrained DNS64 model.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

from src.constants import WavIteratorType
from src.pipeline.noise_injector import NoiseInjector
from src.pipeline.audio_data import AudioData
from src.pipeline.wav import load_wav

TEST_FILE_NAME = "f6581345_nohash_0.wav"
TEST_FILE_PATH = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME}')))
TEST_FILE_OUTPUT_BEFORE_PATH = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME}_before_noise_injection.wav')))
TEST_FILE_OUTPUT_AFTER_PATH = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME}_after_noise_injection.wav')))

def noise_test():
    """
    """
    def transformation_func(audio_data: AudioData) -> AudioData:
        noise_injector = NoiseInjector()
        return noise_injector.transform([audio_data])

    it_transformed = load_wav(TEST_FILE_PATH, 0, WavIteratorType.PLAIN)
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