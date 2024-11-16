"""
Author: Łukasz Kryczka, 2024

This module contains functions to detect speech in an audio signal.
The is_speech function can be used to detect speech in an audio signal
based on different criteria. Currently, only silence detection is supported.
"""
from enum import Enum

import numpy as np

from src.constants import DETECT_SILENCE_THRESHOLD, DETECT_SILENCE_TOLERANCE


class SpeechDetectionType(Enum):
    """
    Enum for different types of speech detection.
    Future types of speech detection can be added here and
    handled in the is_speech function
    """
    SILENCE = 1


def is_speech(audio: np.array, sr: int,
              speech_detection_type: SpeechDetectionType = SpeechDetectionType.SILENCE) -> bool:
    """
    Detect if the audio contains speech.
    @param audio: audio signal
    @param sr: sample rate
    @param speech_detection_type: Type of speech detection to perform
    """

    assert audio.dtype in (np.float32, np.float64)

    if speech_detection_type == SpeechDetectionType.SILENCE:
        return not silence_detection(audio, sr, silence_tolerance=DETECT_SILENCE_TOLERANCE,
                                     silence_threshold=DETECT_SILENCE_THRESHOLD)

    raise ValueError(f"Unsupported speech detection type: {speech_detection_type}")


def silence_detection(
    audio: np.array,
    sr: int,
    silence_tolerance: float,
    silence_threshold : float) -> bool:
    """
    Detect if the audio contains a significant amount of silence.
    
    @param audio: audio signal
    @param sr: sample rate
    @param silence_tolerance: percentage of silence in the audio (0 to 1 range)
    @param silence_threshold: amplitude threshold for silence detection
    @return: True if the audio contains more silence than the silence_tolerance, otherwise False.
    """
    # pylint: disable=unused-argument
    silent_samples = np.sum(np.abs(audio) < silence_threshold)
    total_samples = len(audio)
    silence_percentage = silent_samples / total_samples
    return silence_percentage >= silence_tolerance
