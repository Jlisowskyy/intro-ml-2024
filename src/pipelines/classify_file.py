"""
Author: Jakub Lisowski, 2024

Simple function classifying whole wav data using provided classifier
"""

from typing import Callable

from src.audio.audio_data import AudioData
from src.audio.wav import FlattenWavIterator
from src.constants import MODEL_WINDOW_LENGTH, WavIteratorType


def classify_file(file_path: str, classifier: Callable[[AudioData], int]) -> bool:
    """
    Classify audio data from a file using the provided classifier.

    :param file_path: The path to the audio file to classify
    :param classifier: The classifier function to use

    :return: True if audio belongs 1 claas, False otherwise
    """

    it = FlattenWavIterator(file_path, MODEL_WINDOW_LENGTH, WavIteratorType.OVERLAPPING)
    sr = it.get_first_iter().get_frame_rate()

    is_allowed = False
    for chunk in it:
        audio_data = AudioData(chunk, int(sr))

        print(len(chunk))
        print(5.0 * sr)
        allowed = classifier(audio_data)
        print(allowed)

        is_allowed = is_allowed or allowed

    return is_allowed
