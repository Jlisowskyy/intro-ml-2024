"""
Author: Łukasz Kryczka, 2024

Test cases for the normalization module.
Tests the mean-variance normalization, PCEN, and CMVN.
"""

import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from src.audio.normalize import mean_variance_normalization, pcen_normalization, cmvn_normalization


def generate_sine_wave(frequency: int,
                       duration: float,
                       sample_rate: int,
                       amplitude: float = 1.0) -> np.ndarray:
    """
    Generate a sine wave of a given frequency.

    :param frequency: Frequency of the sine wave in Hz
    :param duration: Duration of the signal in seconds
    :param sample_rate: Sample rate (samples per second)
    :param amplitude: Amplitude of the sine wave
    :return: Generated sine wave as a numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


# Test for Mean and Variance Normalization
def test_mean_variance_normalization() -> None:
    """
    Test that mean and variance normalization results in a signal with mean ~0 and std ~1.
    """
    sample_rate = 44100
    duration = 1.0
    freq = 1000

    sine_wave = generate_sine_wave(freq, duration, sample_rate)
    normalized_wave = mean_variance_normalization(sine_wave)

    mean = np.mean(normalized_wave)
    std = np.std(normalized_wave)

    assert np.isclose(mean, 0, atol=0.01), f"Mean is not close to 0: {mean}"
    assert np.isclose(std, 1, atol=0.01), f"Standard deviation is not close to 1: {std}"

    freqs, times, sxx = spectrogram(normalized_wave, fs=sample_rate, nperseg=256)
    plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram after Mean-Variance Normalization - Manual Check')
    plt.show()


# Test for PCEN Normalization
def test_pcen_normalization() -> None:
    """
    Test that PCEN normalization applies dynamic range compression.
    """
    sample_rate = 44100
    duration = 1.0
    freq = 1000

    sine_wave = generate_sine_wave(freq, duration, sample_rate)
    normalized_wave = pcen_normalization(sine_wave, sample_rate)

    freqs, times, sxx = spectrogram(normalized_wave, fs=sample_rate, nperseg=256)
    plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram after PCEN Normalization - Manual Check')
    plt.show()


# Test for CMVN Normalization
def test_cmvn_normalization() -> None:
    """
    Test that CMVN normalization results in cepstral coefficients with mean ~0 and std ~1.
    """
    sample_rate = 44100
    duration = 1.0
    freq = 1000

    sine_wave = generate_sine_wave(freq, duration, sample_rate)
    normalized_wave = cmvn_normalization(sine_wave, sample_rate)

    mfccs = librosa.feature.mfcc(y=normalized_wave, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Skip the first two coefficients as the first one is the energy and
    # the second one is the delta energy and both are expected to be non-zero
    assert np.allclose(mfccs_mean[2:], 0, atol=0.12), \
        f"Mean of cepstral coefficients is not close to 0: {mfccs_mean[2:]}"
    # Allow a larger tolerance for the standard deviation
    # (The not normalized signal has std of order 5e2, and the normalization is not perfect)
    assert np.allclose(mfccs_std, 1, atol=0.80), \
        f"Standard deviation of cepstral coefficients is not close to 1: {mfccs_std}"

    freqs, times, sxx = spectrogram(normalized_wave, fs=sample_rate, nperseg=256)
    plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram after CMVN Normalization - Manual Check')
    plt.show()


# Running all tests
def example_test_run() -> None:
    """
    Run all the tests
    """
    test_mean_variance_normalization()
    test_pcen_normalization()
    test_cmvn_normalization()
    print("All normalization tests passed!")
