"""
Author: Jakub Lisowski, Michał Kwiatkowski, Tomasz Mycielski, Jakub Pietrzak 2024

Module for classifying audio data using a CNN model.

"""
from sklearn.preprocessing import LabelEncoder

from src.model_definitions import BasicCNN
from src.constants import (MODEL_WINDOW_LENGTH, WavIteratorType,
                           CLASSIFICATION_CONFIDENCE_THRESHOLD, CLASSES)
from src.pipeline.wav import FlattenWavIterator, AudioDataIterator

le = LabelEncoder()
le.fit(CLASSES)


def classify_file(file_path: str, model: BasicCNN) -> str:
    """
    Classify audio data from a file using the provided classifier.

    :param file_path: The path to the audio file to classify
    :param model: The classifier object to use

    :return: The predicted class label or an error message
    """
    it = FlattenWavIterator(file_path, MODEL_WINDOW_LENGTH, WavIteratorType.OVERLAPPING)
    it = AudioDataIterator(it)

    results = [0 for _ in range(len(CLASSES))]
    total_windows = 0

    for audio_data in it:
        predicted_classes = model.classify([audio_data])
        predicted_class = predicted_classes[0]
        if predicted_class < len(CLASSES):
            results[predicted_class] += 1
        total_windows += 1

    if total_windows == 0:
        return "No audio data found in the file"

    max_class_count = max(results)
    predicted_class = results.index(max_class_count)
    confidence = max_class_count / total_windows

    predicted_label = le.inverse_transform([predicted_class])[0]
    print(f"Predicted class: {predicted_label}, confidence: {confidence}")

    if confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
        return "Classification not confident"

    return predicted_label
