"""
Author: Łukasz Kryczka, 2024

Test cases for the speech detection module.
Tests the silence detection feature.
"""

import numpy as np

from src.pipeline.audio_data import AudioData
from src.pipeline.detect_speech import is_speech


def generate_silence(duration: float, sample_rate: int) -> np.ndarray:
    """
    Generate an array of silence (zeros).

    :param duration: Duration of the signal in seconds
    :param sample_rate: Sample rate (samples per second)
    :return: Generated silence as a numpy array
    """
    return np.zeros(int(sample_rate * duration))


def generate_noisy_audio(
        duration: float,
        sample_rate: int,
        noise_amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a noisy audio signal.

    :param duration: Duration of the signal in seconds
    :param sample_rate: Sample rate (samples per second)
    :param noise_amplitude: Amplitude of the noise
    :return: Generated noisy signal as a numpy array
    """
    return noise_amplitude * np.random.randn(int(sample_rate * duration))


# Test for silence detection (pure silence)
def test_silence_detection_pure_silence() -> None:
    """
    Test that silence detection detects pure silence correctly.
    """
    sample_rate = 44100
    duration = 1.0

    silence = generate_silence(duration, sample_rate)
    silence = AudioData(silence, sample_rate)
    is_speech_detected = is_speech(silence)

    assert is_speech_detected is False, "Silence detection failed for pure silence"


# Test for silence detection (noisy signal)
def test_silence_detection_noisy_audio() -> None:
    """
    Test that silence detection correctly identifies non-silent audio.
    """
    sample_rate = 44100
    duration = 1.0

    noisy_audio = generate_noisy_audio(duration, sample_rate)
    noisy_audio = AudioData(noisy_audio, sample_rate)
    is_speech_detected = is_speech(noisy_audio)

    assert is_speech_detected is True, "Silence detection incorrectly identified noise as silence"


# Test for mixed audio (part silence, part noise)
def test_silence_detection_mixed_audio() -> None:
    """
    Test that silence detection correctly identifies silence in mixed audio.
    """
    sample_rate = 44100
    duration = 1.0
    silence_duration = 0.6  # 60% silence
    noise_duration = duration - silence_duration

    silence = generate_silence(silence_duration, sample_rate)
    noise = generate_noisy_audio(noise_duration, sample_rate)

    mixed_audio = np.concatenate((silence, noise))
    mixed_audio = AudioData(mixed_audio, sample_rate)
    is_speech_detected = is_speech(mixed_audio)

    assert is_speech_detected is False, \
        "Silence detection failed for mixed audio with significant silence"

    silence_duration = 0.4  # 40% silence
    noise_duration = duration - silence_duration

    silence = generate_silence(silence_duration, sample_rate)
    noise = generate_noisy_audio(noise_duration, sample_rate)

    mixed_audio = np.concatenate((silence, noise))
    mixed_audio = AudioData(mixed_audio, sample_rate)
    is_speech_detected = is_speech(mixed_audio)

    assert is_speech_detected is True, \
        "Silence detection failed for mixed audio with significant noise"


# Running all tests
def example_test_run() -> None:
    """
    Run all the speech detection tests.
    """
    test_silence_detection_pure_silence()
    test_silence_detection_noisy_audio()
    test_silence_detection_mixed_audio()
    print("All speech detection tests passed!")
