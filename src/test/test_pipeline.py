"""
This module contains tests for the audio processing pipeline, 
including cleaning, normalizing, generating spectrograms, and classifying audio data.
"""

import os
from glob import glob
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
from sklearn.pipeline import Pipeline
from src.pipelines.audio_cleaner import AudioCleaner
from src.pipelines.audio_normalizer import AudioNormalizer
from src.pipelines.spectrogram_generator import SpectrogramGenerator
from src.pipelines.classifier import Classifier
from src.audio.audio_data import AudioData

speaker_to_class = {
    'f1': 1,
    'f7': 1,
    'f8': 1,
    'm3': 1,
    'm6': 1,
    'm8': 1
}

AUDIO_DIRECTORY_PATH = "generated_files"
def gen_random_audio_data():
    """
    Generates random wav files for testing purposes.
    """
    if not os.path.exists(AUDIO_DIRECTORY_PATH):
        os.makedirs(AUDIO_DIRECTORY_PATH)
    for i in range(10):
        audio_data = np.random.randn(18000)
        write(os.path.join(AUDIO_DIRECTORY_PATH, f"random_{i}.wav"), 18000, audio_data)

def example_test_run():
    """
    Example function to run the audio processing pipeline on sample data.
    It creates a training pipeline and fits it on a single audio sample.
    """
    training_pipeline = Pipeline(steps=[
        ('AudioCleaner', AudioCleaner()),
        ('AudioNormalizer', AudioNormalizer()),
        ('SpectrogramGenerator', SpectrogramGenerator()),
        ('Classifier', Classifier())
    ])

    (x_train, y_train) = get_data(AUDIO_DIRECTORY_PATH)

    training_pipeline.fit([x_train[0]], [y_train[0]])
    prediction = training_pipeline.predict([x_train[1]])

    print(prediction)

def get_data(audio_directory_path):
    """
    Retrieves audio data and their corresponding classes from a given directory.

    Args:
        audio_directory_path (str): The path to the directory containing .wav files.

    Returns:
        tuple: A tuple containing a list of AudioData instances and their corresponding labels.
    """
    gen_random_audio_data()
    wav_files = glob(os.path.join(audio_directory_path, "*.wav"))

    x_train = []
    y_train = []

    if not wav_files:
        print("No .wav files found in the directory.")
    else:
        print(f"Found {len(wav_files)} .wav files.")
        for wav_file_path in wav_files:
            audio_data_wav, sample_rate = sf.read(wav_file_path)
            audio_data = AudioData(np.array(audio_data_wav), sample_rate)

            speaker = wav_file_path.split('_')[0]
            speaker_class = speaker_to_class.get(speaker.lower(), 0)

            x_train.append(audio_data)
            y_train.append(speaker_class)

    return (x_train, y_train)
