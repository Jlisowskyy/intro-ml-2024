"""
Author: Jakub Lisowski, 2024

Test the WAV iterator by generating an artificial WAV file and iterating over it.
"""

import os
from contextlib import contextmanager
from typing import Callable

import numpy as np
from scipy.io.wavfile import write

from src.pipeline.audio_data import AudioData
from src.pipeline.wav import load_wav_with_window, AudioDataIterator, FlattenWavIterator
from src.constants import WavIteratorType

AMPLITUDE: float = 0.5
FREQUENCY: int = 960
FILE = "artificial.wav"

# NOTE: AMPLITUDE is set to 0.5, so the generated wave will be in the range [-0.5, 0.5]
CONVERTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "uint8": lambda x: np.uint8((x + AMPLITUDE) * 255),
    "int16": lambda x: np.int16(x * (2 ** 15 - 1)),
    "int32": lambda x: np.int32(x * (2 ** 31 - 1)),
}


def generate_artificial_wav(fps: int, duration: float,
                            np_dtype_converter: Callable[[np.ndarray], np.ndarray]) -> None:
    """
    Generate an artificial WAV file with a single sine wave

    :param fps: Number of frames per second
    :param duration: Duration of the WAV file in seconds
    :param np_dtype_converter: Function to convert the generated wave to the desired NumPy data type

    :return: None
    """

    time = np.linspace(0, duration, int(fps * duration), endpoint=False)
    wave = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * time)

    wave_typed = np_dtype_converter(wave)
    write(FILE, fps, wave_typed)


def remove_artificial_wav() -> None:
    """
    Remove the artificial WAV file

    :return: None
    """

    os.remove(FILE)


@contextmanager
def artificial_wav(fps: int, duration: float, np_dtype: str):
    """
    Context manager for the artificial WAV file

    :param fps: Number of frames per second
    :param duration: Duration of the WAV file in seconds
    :param np_dtype: NumPy data type of the generated wave

    :return: None
    """

    np_dtype_converter = CONVERTERS[np_dtype]

    generate_artificial_wav(fps, duration, np_dtype_converter)
    yield
    remove_artificial_wav()


OVERLAPPING_CASES: list[tuple[int, float, str, float, int]] = [
    (44100, 1.0, "uint8", 0.2, 9),
    (44100, 1.0, "int16", 0.2, 9),
    (44100, 1.0, "int32", 0.2, 9),
    (22050, 1.0, "uint8", 0.2, 9),
    (22050, 1.0, "int16", 0.2, 9),
    (22050, 1.0, "int32", 0.2, 9),
    (44100, 2.0, "uint8", 0.2, 19),
    (44100, 2.0, "int16", 0.2, 19),
    (44100, 2.0, "int32", 0.2, 19),
    (44100, 0, "uint8", 0.2, 0),
]


def test_wav_iter_count_overlapping() -> None:
    """
    Test the number of iterations over the artificial WAV file
    """

    for fps, duration, np_dtype, window_seconds, expected_iterations in OVERLAPPING_CASES:
        with artificial_wav(fps, duration, np_dtype):
            iterator = load_wav_with_window(FILE, window_seconds, 0, WavIteratorType.OVERLAPPING)
            iterations = sum(1 for _ in iterator)

            assert iterations == expected_iterations

    for fps, duration, np_dtype, window_seconds, expected_iterations in OVERLAPPING_CASES:
        with artificial_wav(fps, duration, np_dtype):
            iterator = load_wav_with_window(FILE, window_seconds, 0, WavIteratorType.OVERLAPPING)
            iterator = AudioDataIterator(iterator)
            iterations = sum(1 for _ in iterator)

            assert iterations == expected_iterations


PLAIN_CASES: list[tuple[int, float, str, float, int]] = [
    (44100, 1.0, "uint8", 0.2, 5),
    (44100, 1.0, "int16", 0.2, 5),
    (44100, 1.0, "int32", 0.2, 5),
    (22050, 1.0, "uint8", 0.2, 5),
    (22050, 1.0, "int16", 0.2, 5),
    (22050, 1.0, "int32", 0.2, 5),
    (44100, 2.0, "uint8", 0.2, 10),
    (44100, 2.0, "int16", 0.2, 10),
    (44100, 2.0, "int32", 0.2, 10),
    (44100, 0, "uint8", 0.2, 0),
]


def test_wav_iter_count_plain() -> None:
    """
    Test the number of iterations over the artificial WAV file
    """

    for fps, duration, np_dtype, window_seconds, expected_iterations in PLAIN_CASES:
        with artificial_wav(fps, duration, np_dtype):
            iterator = load_wav_with_window(FILE, window_seconds, 0, WavIteratorType.PLAIN)
            iterations = sum(1 for _ in iterator)

            assert iterations == expected_iterations

    for fps, duration, np_dtype, window_seconds, expected_iterations in PLAIN_CASES:
        with artificial_wav(fps, duration, np_dtype):
            iterator = load_wav_with_window(FILE, window_seconds, 0, WavIteratorType.PLAIN)
            iterator = AudioDataIterator(iterator)
            iterations = sum(1 for _ in iterator)

            assert iterations == expected_iterations


def test_audio_data_returns_same() -> None:
    """
    Test if the AudioData object returns the same audio signal as the raw audio data
    """

    for fps, duration, np_dtype, window_seconds, _ in PLAIN_CASES:
        with artificial_wav(fps, duration, np_dtype):
            iterator = load_wav_with_window(FILE, window_seconds, 0, WavIteratorType.PLAIN)
            iterator1 = AudioDataIterator(iterator)

            for _, (audio_data1, audio_data2) in enumerate(zip(iterator, iterator1)):
                assert np.array_equal(
                    AudioData(audio_data1, int(iterator.get_frame_rate())).audio_signal,
                    audio_data2.audio_signal)

    for fps, duration, np_dtype, window_seconds, _ in PLAIN_CASES:
        with artificial_wav(fps, duration, np_dtype):
            iterator = FlattenWavIterator(FILE, window_seconds, WavIteratorType.PLAIN)
            iterator1 = AudioDataIterator(iterator)

            for _, (audio_data1, audio_data2) in enumerate(zip(iterator, iterator1)):
                assert np.array_equal(AudioData(
                    audio_data1,
                    int(iterator.get_first_iter().get_frame_rate())).audio_signal,
                                      audio_data2.audio_signal)
