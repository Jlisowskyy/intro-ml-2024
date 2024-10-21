"""
Author: Micał Kwiatkowski

This module processes audio data through a pipeline that cleans, normalizes, and generates mel 
spectrograms. The pipeline uses `AudioCleaner`, `AudioNormalizer`, and `SpectrogramGenerator` 
to transform the audio and save spectrograms as images.
"""

import os
from glob import glob
import soundfile as sf
import numpy as np
from sklearn.pipeline import Pipeline
from src.pipelines.audio_cleaner import AudioCleaner
from src.pipelines.audio_normalizer import AudioNormalizer
# pylint: disable=line-too-long
from src.pipelines.spectrogram_generator import SpectrogramGenerator, gen_mel_spectrogram, save_spectrogram
from src.audio.audio_data import AudioData

speaker_to_class = {
    'f1': 1,
    'f7': 1,
    'f8': 1,
    'm3': 1,
    'm6': 1,
    'm8': 1
}

AUDIO_DIRECTORY_PATH = "/home/michal/studia/sem5/ml/daps/clean"
SPECTROGRAM_CLEANED_PATH = "/home/michal/studia/sem5/ml/spec_clean.png"
SPECTROGRAM_PATH = "/home/michal/studia/sem5/ml/spec.png"

def example_test_run():
    """
    Run an example of the transformation pipeline on the first audio file in the dataset.
    
    The pipeline includes cleaning, normalizing, and generating spectrograms. It saves both
    the cleaned and uncleaned spectrograms.
    """
    transformation_pipeline = Pipeline(steps=[
        ('AudioCleaner', AudioCleaner()),
        ('AudioNormalizer', AudioNormalizer()),
        ('SpectrogramGenerator', SpectrogramGenerator())
    ])

    (x_train, _) = get_data(AUDIO_DIRECTORY_PATH)

    # Transformed data
    transformation_pipeline.fit([x_train[0]])
    model_input = transformation_pipeline.transform([x_train[0]])

    save_spectrogram(model_input[0], SPECTROGRAM_CLEANED_PATH)

    spectrogram_not_cleaned = gen_mel_spectrogram(x_train[0].audio_signal, x_train[0].sample_rate)
    save_spectrogram(spectrogram_not_cleaned, SPECTROGRAM_PATH)

def get_data(audio_directory_path):
    """
    Retrieves audio data and their corresponding classes from a given directory.

    Args:
        audio_directory_path (str): The path to the directory containing .wav files.

    Returns:
        tuple: A tuple containing a list of AudioData instances and their corresponding labels.
    """
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
