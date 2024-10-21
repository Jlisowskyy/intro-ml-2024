"""
Author: Michał Kwiatkowski

This module runs an example of the audio processing pipeline on a sample audio file. It reads the
audio, converts it into an `AudioData` object, and uses the `classify_wrapper` function to predict
the class of the audio.
"""

import soundfile as sf
import numpy as np
from src.pipelines.classify import classify_wrapper
from src.audio.audio_data import AudioData

AUDIO_FILE_PATH = "/home/michal/Downloads/f2_script1_ipad_office1_35000.wav"

def example_test_run():
    """
    Example function to run the audio processing pipeline on sample data.
    It creates a training pipeline and fits it on a single audio sample.
    """
    audio_data_wav, sample_rate = sf.read(AUDIO_FILE_PATH)
    audio_data = AudioData(np.array(audio_data_wav), sample_rate)

    prediction = classify_wrapper(audio_data)

    print(prediction)
