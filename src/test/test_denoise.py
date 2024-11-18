"""
Author: Łukasz Kryczka, 2024

Test cases for the denoise module.
Currently, tests the basic denoising filter.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import spectrogram

from src.constants import DenoiseType
from src.pipelines.audio_cleaner import AudioCleaner

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


def save_wave(file_name: str, data: np.ndarray, sample_rate: int) -> None:
    """
    Save a numpy array as a WAV file.

    :param file_name: File name to save as
    :param data: Audio data
    :param sample_rate: Sample rate of the data
    """

    write(file_name, sample_rate, data)


def load_wave(file_name: str) -> tuple[int, np.ndarray]:
    """
    Load a WAV file and return its sample rate and data.

    :param file_name: File name to load
    :return: Sample rate and audio data as numpy array
    """

    return read(file_name)


# Actual test cases

def test_denoise_basic_low_freq_filtering() -> None:
    """
    Test that frequencies below 50 Hz are reduced by the denoise_basic filter.
    """

    sample_rate = 44100
    duration = 1.0
    low_freq = 20

    sine_wave = generate_sine_wave(low_freq, duration, sample_rate)
    filtered_wave = AudioCleaner.denoise(sine_wave, sample_rate, DenoiseType.BASIC)

    assert np.max(np.abs(filtered_wave)) < 0.10, "Low frequencies were not properly reduced"


def test_denoise_basic_high_freq_filtering() -> None:
    """
    Test that frequencies above 8500 Hz are reduced by the denoise_basic filter.
    """

    sample_rate = 44100
    duration = 1.0
    high_freq = 16500

    sine_wave = generate_sine_wave(high_freq, duration, sample_rate)
    filtered_wave = AudioCleaner.denoise(sine_wave, sample_rate, DenoiseType.BASIC)

    assert np.max(np.abs(filtered_wave)) < 0.10, "High frequencies were not properly reduced"

def manual_test_denoise_basic_passband_freq() -> None:
    """
    Test that frequencies within the passband (100 Hz - 8000 Hz)
    are preserved by the denoise_basic filter.
    """

    sample_rate = 44100
    duration = 1.0
    passband_freq = 2000

    sine_wave = generate_sine_wave(passband_freq, duration, sample_rate)
    filtered_wave = AudioCleaner.denoise(sine_wave, sample_rate, DenoiseType.BASIC)

    # Before
    freqs, times, sxx = spectrogram(sine_wave, fs=sample_rate, nperseg=256)
    plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram before Denoising Passband Frequencies - Manual Check')
    plt.show()

    # After
    freqs, times, sxx = spectrogram(filtered_wave, fs=sample_rate, nperseg=256)
    plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram after Denoising Passband Frequencies - Manual Check')
    plt.show()

    # NOTE: Investigate
    # ??? The results are not as expected and surprising

    assert np.allclose(sine_wave, filtered_wave, atol=0.20), \
        "Passband frequencies were not preserved"

def manual_test_denoise_basic_mixed_freq() -> None:
    """
    Test that a mixture of frequencies is correctly filtered by the denoise_basic filter.
    """

    sample_rate = 44100
    duration = 1.0
    low_freq = 50
    passband_freq1 = 500
    passband_freq2 = 3000
    high_freq = 12000

    mixed_wave = (generate_sine_wave(low_freq, duration, sample_rate) +
                  generate_sine_wave(passband_freq1, duration, sample_rate) +
                  generate_sine_wave(passband_freq2, duration, sample_rate) +
                  generate_sine_wave(high_freq, duration, sample_rate))

    filtered_wave = AudioCleaner.denoise(mixed_wave, sample_rate, DenoiseType.BASIC)

    # Manual check of the spectrogram
    freqs, times, sxx = spectrogram(filtered_wave, fs=sample_rate, nperseg=256)
    plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram after Denoising Mixed Frequencies - Manual Check')
    plt.show()


# Running all tests
def manual_test() -> None:
    """
    Run all the tests
    """

    manual_test_denoise_basic_passband_freq()
    manual_test_denoise_basic_mixed_freq()
