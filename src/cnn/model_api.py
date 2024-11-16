"""
Author: Jakub Lisowski, Michal Kwiatkowski, Tomasz Mycielski, Jakub Pietrzak 2024

Module for classifying audio data using a CNN model.

"""

from src.audio.wav import FlattenWavIterator, AudioDataIterator
from src.cnn.cnn import BasicCNN
from src.constants import MODEL_WINDOW_LENGTH, WavIteratorType

def classify_file(file_path: str, model: BasicCNN) -> bool:
    """
    Classify audio data from a file using the provided classifier.

    :param file_path: The path to the audio file to classify
    :param model: The classifier object to use

    :return: True if audio belongs 1 claas, False otherwise
    """

    it = FlattenWavIterator(file_path, MODEL_WINDOW_LENGTH, WavIteratorType.OVERLAPPING)
    it = AudioDataIterator(it)

    results = [0, 0]
    for chunk in it:
        result = model.classify(chunk)
        print(f"Classified as: {result}")
        results[result] += 1

    total = results[0] + results[1]
    return results[1] / total > 0.7
