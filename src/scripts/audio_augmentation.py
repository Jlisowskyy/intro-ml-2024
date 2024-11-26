"""
Author: Jakub Pietrzak, 2024

Script for augmenting audio files with various effects.
"""

import argparse
import numpy as np

import soundfile as sf
from scipy.signal import convolve

from src.pipeline.audio_data import AudioData


DEFAULT_SEMITONES = 6
DEFAULT_SPEED_FACTOR = 0.5
DEFAULT_NOISE_LEVEL = 0.03
DEFAULT_GAIN_DB = -10
DEFAULT_REVERB_AMOUNT = 0.1
DEFAULT_ECHO_DELAY = 0.25
DEFAULT_ECHO_DECAY = 0.6

def load_file(file_path: str) -> AudioData:
    """
    Loads an audio file from the specified path into an AudioData object.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        AudioData: The audio data loaded from the file.
    """
    audio_data, sample_rate = sf.read(file_path)
    return AudioData(audio_data, sample_rate)


def save_file(audio_data: AudioData, file_path: str) -> None:
    """
    Saves the given audio data to the specified file path.

    Args:
        audio_data (AudioData): The audio data to save.
        file_path (str): The path where the audio file should be saved.
    """
    sf.write(file_path, audio_data.audio_signal, audio_data.sample_rate)


def change_pitch(audio_data: AudioData, semitones: float) -> AudioData:
    """
    Changes the pitch of the audio by the given number of semitones.

    Args:
        audio_data (AudioData): The original audio data.
        semitones (float): The number of semitones to shift the pitch.

    Returns:
        AudioData: The pitch-shifted audio data.
    """
    factor = 2 ** (semitones / 12)
    indices = np.round(np.arange(0, len(audio_data.audio_signal), factor))
    indices = indices[indices < len(audio_data.audio_signal)].astype(int)
    new_signal = audio_data.audio_signal[indices]
    return AudioData(new_signal, audio_data.sample_rate)


def change_speed(audio_data: AudioData, speed_factor: float) -> AudioData:
    """
    Changes the speed of the audio (faster or slower).

    Args:
        audio_data (AudioData): The original audio data.
        speed_factor (float): The factor by which to speed up or slow down the audio.

    Returns:
        AudioData: The speed-changed audio data.
    """
    indices = np.round(np.arange(0, len(audio_data.audio_signal), 1 / speed_factor))
    indices = indices[indices < len(audio_data.audio_signal)].astype(int)
    new_signal = audio_data.audio_signal[indices]
    return AudioData(new_signal, audio_data.sample_rate)


def add_noise(audio_data: AudioData, noise_level: float) -> AudioData:
    """
    Adds random noise to the audio.

    Args:
        audio_data (AudioData): The original audio data.
        noise_level (float): The standard deviation of the noise to add.

    Returns:
        AudioData: The audio data with added noise.
    """
    noise = np.random.normal(0, noise_level, audio_data.audio_signal.shape)
    noisy_signal = audio_data.audio_signal + noise
    return AudioData(noisy_signal, audio_data.sample_rate)


def change_volume(audio_data: AudioData, gain_db: float) -> AudioData:
    """
    Changes the volume of the audio by a given decibel level.

    Args:
        audio_data (AudioData): The original audio data.
        gain_db (float): The gain to apply in dB (positive for increase, negative for decrease).

    Returns:
        AudioData: The volume-adjusted audio data.
    """
    new_signal = audio_data.audio_signal * (10 ** (gain_db / 20))
    return AudioData(new_signal, audio_data.sample_rate)


def add_reverb(audio_data: AudioData, reverb_amount: float = 0.3) -> AudioData:
    """
    Adds a simple reverb effect to the audio.

    Args:
        audio_data (AudioData): The original audio data.
        reverb_amount (float): The amount of reverb to apply (default is 0.3).

    Returns:
        AudioData: The audio data with reverb.
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

    Args:
        audio_data (AudioData): The original audio data.
        delay (float): The delay of the echo in seconds.
        decay (float): The decay rate of the echo (how much the echo fades).

    Returns:
        AudioData: The audio data with echo.
    """
    delay_samples = int(audio_data.sample_rate * delay)
    echo_signal = np.zeros(len(audio_data.audio_signal) + delay_samples)
    echo_signal[: len(audio_data.audio_signal)] = audio_data.audio_signal
    for i, sample in enumerate(audio_data.audio_signal):
        echo_signal[i + delay_samples] += sample * decay
    echo_signal = echo_signal[: len(audio_data.audio_signal)]
    return AudioData(echo_signal, audio_data.sample_rate)


def augmentations(audio_data: AudioData, options: list,
                  semitones=DEFAULT_SEMITONES,
                  speed_factor=DEFAULT_SPEED_FACTOR,
                  noise_level=DEFAULT_NOISE_LEVEL,
                  gain_db=DEFAULT_GAIN_DB,
                  reverb_amount=DEFAULT_REVERB_AMOUNT,
                  echo_delay=DEFAULT_ECHO_DELAY,
                  echo_decay=DEFAULT_ECHO_DECAY) -> AudioData:
    """
    Applies a series of augmentations to the given audio data based on the specified options.

    Args:
        audio_data (AudioData): The original audio data.
        options (list): A list of augmentation options to apply. Possible values are:
                        ['pitch', 'speed', 'noise', 'volume', 'reverb', 'echo'].
        semitones (int): The number of semitones to shift the pitch (default is 6).
        speed_factor (float): The factor by which to change the speed (default is 0.5).
        noise_level (float): The level of noise to add (default is 0.10).
        gain_db (float): The gain in decibels to adjust the volume (default is -10).
        reverb_amount (float): The amount of reverb to apply (default is 0.1).
        echo_delay (float): The delay in seconds before the echo (default is 0.25).
        echo_decay (float): The decay rate of the echo (default is 0.6).

    Returns:
        AudioData: The augmented audio data.
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


def process(file_path: str, output_path: str) -> None:
    """
    Processes the audio file by applying augmentations based on the options and saves the results.

    Args:
        file_path (str): The path to the input audio file.
        output_path (str): The base path for saving the output audio files.
    """
    audio_data = load_file(file_path)
    options = ['pitch', 'speed', 'noise', 'volume', 'reverb', 'echo']

    for option in options:
        audio_data_augmented = augmentations(audio_data, [option])
        new_output_path = output_path.split('.')[0] + "_" + option + '.' + output_path.split('.')[1]
        save_file(audio_data_augmented, new_output_path)


def main(argv: list[str]) -> None:
    """
    Main function that parses command line arguments and runs the processing.

    Args:
        argv (List[str]): List of command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generates and saves augmented versions of an audio file.")
    parser.add_argument(
        "--file_path", '-f', type=str, required=True, help="Path to the input audio file.")
    parser.add_argument(
        "--output_path", '-o', type=str, required=True, help="Path to save the output audio file.")

    args = parser.parse_args(argv)
    process(args.file_path, args.output_path)
