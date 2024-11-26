"""
Author: Michał Kwiatkowski
"""

from os import listdir, path
from random import choice
import numpy as np
import soundfile as sf
from src.constants import DATABASE_OUT_NOISES
from src.pipeline.audio_data import AudioData

class NoiseInjector:
    """
    ni ma
    """

    def __init__(self, noise_folder_path = DATABASE_OUT_NOISES) -> None:
        self.noise_folder_path = noise_folder_path
        return

    @staticmethod
    def get_random_audio_file(folder_path: str) -> str:
        """
        Get a random audio file from a specified folder.

        Args:
            folder_path (str): The path to the folder containing audio files.

        Returns:
            str: The path to a randomly selected audio file.
        """
        # List all files in the folder
        files = listdir(folder_path)

        # Filter the list to include only audio files
        audio_files = [f for f in files if f.endswith('.wav')]

        if not audio_files:
            raise ValueError("No audio files found in the specified folder.")

        # Select a random file
        random_file = choice(audio_files)
        return path.join(folder_path, random_file)

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'NoiseInjector':
        """
        ni ma
        """
        return

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Inject random noise from a noise file to the input audio data.
        
        Args:
            x_data (list[AudioData]): List of input audio data to be noise-injected.
        
        Returns:
            list[AudioData]: List of noise-injected audio data.
        """
        # Noise-injected audio list
        noisy_audio_data = []

        for audio_data in x_data:
            # Load the noise file
            noise_file_path = NoiseInjector.get_random_audio_file(self.noise_folder_path)
            noise_audio, noise_sample_rate = sf.read(noise_file_path)

            # Convert noise to float32 using AudioData's to_float method
            noise_audio = AudioData.to_float(noise_audio)
            
            if noise_sample_rate != audio_data.sample_rate:
                # Resample noise if sample rates don't match
                raise ValueError(f"Sample rate mismatch: noise {noise_sample_rate} vs audio {audio_data.sample_rate}")

            if len(noise_audio) < len(audio_data.audio_signal):
                # Repeat noise to match audio length
                repeats = int(np.ceil(len(audio_data.audio_signal) / len(noise_audio)))
                noise_audio = np.tile(noise_audio, repeats)

            # Trim noise to match audio length
            noise_segment = noise_audio[:len(audio_data.audio_signal)]

            # Add noise with a random SNR
            snr = np.random.uniform(10, 30)
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
