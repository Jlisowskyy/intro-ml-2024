"""
Author: Łukasz Kryczka, 2024

Test cases for the normalization module.
Tests the mean-variance normalization, PCEN, and CMVN using WAV file input.
"""

import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import spectrogram
import soundfile as sf

from src.audio.normalize import mean_variance_normalization, pcen_normalization, cmvn_normalization

# Define paths
TEST_DATA_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/f2_script1_ipad_office1_35000.wav')))


def load_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return the audio data and sample rate.

    :param file_path: Path to the audio file
    :return: Tuple of (audio_data, sample_rate)
    """
    audio_data, sample_rate = sf.read(file_path)
    return audio_data, sample_rate


def plot_spectrograms(original: np.ndarray,
                     normalized: np.ndarray,
                     sample_rate: int,
                     title: str) -> None:
    """
    Plot spectrograms of original and normalized signals side by side.

    :param original: Original signal
    :param normalized: Normalized signal
    :param sample_rate: Sample rate of the signals
    :param title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Original spectrogram
    freqs1, times1, sxx1 = spectrogram(original, fs=sample_rate, nperseg=256)
    pcm1 = ax1.pcolormesh(times1, freqs1, 10 * np.log10(sxx1), shading='gouraud')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [s]')
    ax1.set_title('Original Signal')
    fig.colorbar(pcm1, ax=ax1, label='Power [dB]')

    # Normalized spectrogram
    freqs2, times2, sxx2 = spectrogram(normalized, fs=sample_rate, nperseg=256)
    pcm2 = ax2.pcolormesh(times2, freqs2, 10 * np.log10(sxx2), shading='gouraud')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Normalized Signal')
    fig.colorbar(pcm2, ax=ax2, label='Power [dB]')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def manual_test_mean_variance_normalization() -> None:
    """
    Test that mean and variance normalization results in a signal with mean ~0 and std ~1.
    """
    audio_data, sample_rate = load_audio_file(TEST_DATA_PATH)
    normalized_wave = mean_variance_normalization(audio_data)

    mean = np.mean(normalized_wave)
    std = np.std(normalized_wave)

    assert np.isclose(mean, 0, atol=0.01), f"Mean is not close to 0: {mean}"
    assert np.isclose(std, 1, atol=0.01), f"Standard deviation is not close to 1: {std}"

    plot_spectrograms(
        audio_data,
        normalized_wave,
        sample_rate,
        'Mean-Variance Normalization Comparison'
    )


def manual_test_pcen_normalization() -> None:
    """
    Test that PCEN normalization applies dynamic range compression.
    """
    audio_data, sample_rate = load_audio_file(TEST_DATA_PATH)
    normalized_wave = pcen_normalization(audio_data, sample_rate)

    plot_spectrograms(
        audio_data,
        normalized_wave,
        sample_rate,
        'PCEN Normalization Comparison'
    )


def manual_test() -> None:
    """
    Run all the tests
    """
    manual_test_mean_variance_normalization()
    manual_test_pcen_normalization()


if __name__ == "__main__":
    manual_test()