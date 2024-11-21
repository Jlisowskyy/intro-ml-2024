"""
Author: Łukasz Kryczka, 2024

Module contains functions to detect speech or silence in the audio signal.
"""

import numpy as np

from src.constants import DETECT_SILENCE_THRESHOLD, DETECT_SILENCE_TOLERANCE
from src.pipeline.audio_data import AudioData


def is_speech(audio_data: AudioData, silence_tolerance: float = DETECT_SILENCE_TOLERANCE,
              silence_threshold: float = DETECT_SILENCE_THRESHOLD) -> bool:
    """
    Detects if there is speech in the audio signal.

    :param audio_data: audio signal
    :param silence_tolerance: maximum percentage of silence in the signal
    :param silence_threshold: threshold for silence detection

    :return: True if speech is detected, False otherwise
    """

    assert audio_data.audio_signal.dtype in (np.float32, np.float64)

    silent_samples = np.sum(np.abs(audio_data.audio_signal) < silence_threshold)
    total_samples = len(audio_data.audio_signal)
    silence_percentage = silent_samples / total_samples
    return not silence_percentage >= silence_tolerance
