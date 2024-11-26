"""
Author: Michał Kwiatkowski, 2024

This file defines a NoiseInjector class that injects random noise from a specified folder of 
audio files into a list of input audio data. The noise is scaled based on a randomly chosen 
signal-to-noise ratio (SNR) between a specified lower and upper bound.

The NoiseInjector class includes methods to load random audio files, apply noise injection to 
audio signals, and handle potential mismatches in sample rates and audio lengths. The class is 
designed to process lists of AudioData objects and return a list of noise-injected AudioData 
objects.
"""

from os import listdir, path
from random import Random
import numpy as np
import soundfile as sf
from src.constants import DATABASE_OUT_NOISES, DEFAULT_SEED, SNR_BOTTOM_BOUND, SNR_UPPER_BOUND
from src.pipeline.audio_data import AudioData

class NoiseInjector:
    """
    The NoiseInjector class injects random noise from a folder of noise files into a list of 
    input audio data, modifying the audio signals by adding scaled noise with a randomly 
    selected signal-to-noise ratio (SNR).

    Attributes:
        noise_folder_path (str): Path to the folder containing noise audio files. Default is 
                                  `DATABASE_OUT_NOISES`.
    """

    def __init__(self, noise_folder_path = DATABASE_OUT_NOISES, seed: int = DEFAULT_SEED) -> None:
        """
        Initializes the NoiseInjector with the path to the folder containing noise files.

        Parameters:
            noise_folder_path (str): Path to the folder with noise audio files. Defaults to 
                                      `DATABASE_OUT_NOISES`.
        """
        self.noise_folder_path = noise_folder_path
        self._seed = seed
        self._rng = Random(seed)

    def get_random_audio_file(self, folder_path: str) -> str:
        """
        Get a random audio file from a specified folder.

        Parameters:
            folder_path (str): The path to the folder containing audio files.

        Returns:
            str: The path to a randomly selected audio file.

        Raises:
            ValueError: If no audio files are found in the specified folder.
        """
        files = listdir(folder_path)
        audio_files = [f for f in files if f.endswith('.wav')]

        if not audio_files:
            raise ValueError("No audio files found in the specified folder.")

        random_file = self._rng.choice(audio_files)
        return path.join(folder_path, random_file)

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'NoiseInjector':
        """
        Placeholder method for fitting the noise injector, which is not implemented.
        This function does not perform any actions in the current version.

        Parameters:
            x_data (list[AudioData]): List of input audio data to fit.
            y_data (list[int], optional): List of target labels corresponding to the input audio 
                                          data.

        Returns:
            NoiseInjector: The instance of the NoiseInjector class (unchanged).
        """
        return

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Injects random noise into the input audio data.

        Parameters:
            x_data (list[AudioData]): List of input audio data to be noise-injected.

        Returns:
            list[AudioData]: List of audio data with added noise.

        Raises:
            ValueError: If the sample rate of the noise does not match the sample rate of the 
                        input audio.
        """
        noisy_audio_data = []

        for audio_data in x_data:
            # Load the noise file
            noise_file_path = self.get_random_audio_file(self.noise_folder_path)
            noise_audio, noise_sample_rate = sf.read(noise_file_path)

            noise_audio = AudioData.to_float(noise_audio)

            if noise_sample_rate != audio_data.sample_rate:
                raise ValueError(f"Sample rate mismatch: noise {noise_sample_rate} vs audio "
                                 f"{audio_data.sample_rate}")

            if len(noise_audio) < len(audio_data.audio_signal):
                # Repeat noise to match audio length
                repeats = int(np.ceil(len(audio_data.audio_signal) / len(noise_audio)))
                noise_audio = np.tile(noise_audio, repeats)

            # Trim noise to match audio length
            noise_segment = noise_audio[:len(audio_data.audio_signal)]

            # Add noise with a random SNR
            snr = np.random.uniform(SNR_BOTTOM_BOUND, SNR_UPPER_BOUND)
            noise_power = np.mean(noise_segment**2)
            signal_power = np.mean(audio_data.audio_signal**2)
            noise_scale = np.sqrt(signal_power / (noise_power * (10 ** (snr/10))))

            # Mix original audio with scaled noise
            noisy_signal = audio_data.audio_signal + noise_scale * noise_segment

            # Clip to prevent overflow and maintain float32 range
            noisy_signal = np.clip(noisy_signal, -1.0, 1.0)

            # Create new AudioData with noisy signal
            noisy_audio = AudioData(noisy_signal, audio_data.sample_rate)
            noisy_audio_data.append(noisy_audio)

        return noisy_audio_data
