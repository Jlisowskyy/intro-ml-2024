"""
Author: Jakub Pietrzak, 2024

Modul for generating spectrograms and showing/saving it
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.pipeline import Pipeline
from src.pipeline.audio_data import AudioData

from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.scripts.spectrogram_from_npy import get_random_file_path
from scipy.signal import convolve
from pydub import AudioSegment
from pydub.effects import speedup


def load_file(file_path: str) -> AudioData:
    audio_data, sample_rate = sf.read(file_path)
    return AudioData(audio_data, sample_rate)

def save_file(audio_data: AudioData, file_path: str) -> None:
    sf.write(file_path, audio_data.audio_signal, audio_data.sample_rate)
    

def change_pitch(audio_data: AudioData, semitones: float) -> AudioData:
    """
    Changes the pitch of the audio by the given number of semitones.
    """
    factor = 2 ** (semitones / 12)
    indices = np.round(np.arange(0, len(audio_data.audio_signal), factor))
    indices = indices[indices < len(audio_data.audio_signal)].astype(int)
    new_signal = audio_data.audio_signal[indices]
    return AudioData(new_signal, audio_data.sample_rate)


def change_speed(audio_data: AudioData, speed_factor: float) -> AudioData:
    """
    Changes the speed of the audio (faster or slower).
    """
    indices = np.round(np.arange(0, len(audio_data.audio_signal), 1 / speed_factor))
    indices = indices[indices < len(audio_data.audio_signal)].astype(int)
    new_signal = audio_data.audio_signal[indices]
    return AudioData(new_signal, audio_data.sample_rate)


def add_noise(audio_data: AudioData, noise_level: float) -> AudioData:
    """
    Adds random noise to the audio.
    """
    noise = np.random.normal(0, noise_level, audio_data.audio_signal.shape)
    noisy_signal = audio_data.audio_signal + noise
    return AudioData(noisy_signal, audio_data.sample_rate)


def change_volume(audio_data: AudioData, gain_db: float) -> AudioData:
    """
    Changes the volume of the audio by a given decibel level.
    """
    new_signal = audio_data.audio_signal * (10 ** (gain_db / 20))
    return AudioData(new_signal, audio_data.sample_rate)


def add_reverb(audio_data: AudioData, reverb_amount: float = 0.3) -> AudioData:
    """
    Adds a simple reverb effect to the audio.
    """
    impulse_length = int(audio_data.sample_rate * reverb_amount)  
    impulse_response = np.zeros(impulse_length)
    
    impulse_response[0] = 1.0  
    impulse_response[-1] = 0.5  
    impulse_response[1:] = np.linspace(1.0, 0.5, impulse_length - 1)
    reverberated_signal = convolve(audio_data.audio_signal, impulse_response, mode='full')
    reverberated_signal = reverberated_signal[:len(audio_data.audio_signal)]
    return AudioData(reverberated_signal, audio_data.sample_rate)


def add_echo(audio_data: AudioData, delay: float = 0.2, decay: float = 0.5) -> AudioData:
    """
    Adds an echo effect to the audio.
    """
    delay_samples = int(audio_data.sample_rate * delay)
    echo_signal = np.zeros(len(audio_data.audio_signal) + delay_samples)
    echo_signal[: len(audio_data.audio_signal)] = audio_data.audio_signal
    for i in range(len(audio_data.audio_signal)):
        echo_signal[i + delay_samples] += audio_data.audio_signal[i] * decay
    echo_signal = echo_signal[: len(audio_data.audio_signal)]  # Trim back to original length
    return AudioData(echo_signal, audio_data.sample_rate)



def augmentations(audio_data: AudioData, options:list , semitones=6, speed_factor=0.5,
                  noise_level=0.10, gain_db=-10, reverb_amount=0.1, echo_delay=0.25,
                  echo_decay=0.6) -> AudioData:
    """
    Applies a chain of augmentations to the audio data.
    
    options = ['pitch', 'speed', 'noise', 'volume', 'reverb', 'echo']
    """
    if 'pitch' in options:
        audio_data = change_pitch(audio_data, semitones)
    
    if 'speed' in options:
        audio_data = change_speed(audio_data, speed_factor)
    
    if 'noise' in options:
        audio_data = add_noise(audio_data, noise_level)
    
    if 'volume' in options:
        audio_data = change_volume(audio_data, gain_db)
    
    if 'reverb' in options:
        audio_data = add_reverb(audio_data, reverb_amount)
    
    if 'echo' in options:
        audio_data = add_echo(audio_data, echo_delay, echo_decay)
    return audio_data


   

def process(file_path:str, output_path:str) -> None:
    
    audio_data = load_file(file_path)
    options = ['pitch', 'speed', 'noise', 'volume', 'reverb', 'echo']
    for option in options:
        audio_data_augmented = augmentations(audio_data, [option])
        new_output_path=output_path.split('.')[0] + option + '.' + output_path.split('.')[1]
        save_file(audio_data_augmented, new_output_path)
    
    

def main(argv: list[str]) -> None:
    """
    Main function that parses command line arguments and runs the processing.

    Args:
        argv (List[str]): List of command line arguments.
    """
    
   
    parser = argparse.ArgumentParser(description="Generates a spectrogram from an audio file.")
    parser.add_argument("--file_path", '-f', type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--output_path", '-o', type=str, required=True, help="Path to the output audio file.")

    args = parser.parse_args(argv)
    process(args.file_path, args.output_path)

import sys
if __name__ == "__main__":
    main(sys.argv[1:])