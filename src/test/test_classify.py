"""
Author: Michał Kwiatkowski
This module runs an example of the audio processing pipeline on a sample audio file. It reads the
audio, converts it into an `AudioData` object, and uses the `classify_wrapper` function to predict
the class of the audio.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from src.cnn.cnn import BasicCNN
from src.pipeline.audio_data import AudioData

# TODO: Kill this file

# pylint: disable=line-too-long
TEST_FILE_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f2_script1_ipad_office1_35000.wav')))
MODEL_PATH = "/home/wookie/Projects/intro-ml-2024/cnn_e9_backup.pth"

def example_test_run():
    """
    Example function to run the audio processing pipeline on sample data.
    It creates a training pipeline and fits it on a single audio sample.
    """
    audio_data_wav, sample_rate = sf.read(TEST_FILE_PATH)
    audio_data = AudioData(np.array(audio_data_wav), sample_rate)

    classifier = BasicCNN.load_model(MODEL_PATH)

    print(classifier.classify([audio_data]))
