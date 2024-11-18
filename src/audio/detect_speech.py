"""
Author: Łukasz Kryczka, 2024

Simple speech detection module.
"""

import numpy as np

from src.constants import DETECT_SILENCE_THRESHOLD, DETECT_SILENCE_TOLERANCE


def is_speech(audio: np.ndarray, silence_tolerance: float = DETECT_SILENCE_TOLERANCE,
              silence_threshold: float = DETECT_SILENCE_THRESHOLD) -> bool:
    """
    Detects if there is speech in the audio signal.

    :param audio: audio signal
    :param silence_tolerance: maximum percentage of silence in the signal
    :param silence_threshold: threshold for silence detection

    :return: True if speech is detected, False otherwise
    """

    assert audio.dtype in (np.float32, np.float64)

    silent_samples = np.sum(np.abs(audio) < silence_threshold)
    total_samples = len(audio)
    silence_percentage = silent_samples / total_samples
    return not silence_percentage >= silence_tolerance
