"""
Author: Michał Kwiatkowski, Tomasz Mycielski

Module for classifying audio data using a CNN model.

This module contains the classify function which processes audio data
through a series of transformations and passes it to a CNN model for prediction.
"""

import torch

from src.audio.audio_data import AudioData
from src.audio.denoise import denoise
from src.audio.normalize import normalize
from src.audio.spectrogram import gen_mel_spectrogram
from src.cnn.cnn import BasicCNN
from src.constants import NORMALIZATION_TYPE, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT


def classify(audio_data: AudioData, model: BasicCNN) -> int:
    """
    Classify audio data using the provided CNN model.

    Args:
        data (AudioData): The audio data to classify.
        model (BasicCNN): The CNN model used for classification.

    Returns:
        int: user's class.
    """

    sr = audio_data.sample_rate
    print(sr)
    chunk = audio_data.audio_signal
    chunk = denoise(chunk, sr)
    chunk = normalize(chunk, sr, NORMALIZATION_TYPE)
    spectrogram = gen_mel_spectrogram(chunk, int(sr),
                                      width=SPECTROGRAM_WIDTH,
                                      height=SPECTROGRAM_HEIGHT)
    tens = torch.from_numpy(spectrogram).type(torch.float32)
    tens = torch.rot90(tens, dims=(0, 2))
    tens = tens[None, :, :, :]
    print(tens.shape)

    # checking device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    tens.to(device)

    # prediction
    with torch.no_grad():
        prediction = model(tens)
    return prediction[0].argmax(0).item()
