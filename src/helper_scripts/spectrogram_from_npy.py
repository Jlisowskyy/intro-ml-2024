"""
Author: Jakub Pietrzak, 2024

Script for generating spectrograms form npy files and showing them
"""

import os
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np


def get_random_npy_path(dir_path: str) -> str:
    """
    Gets a random audio file path from a directory.
    Args:
        dir_path (str): path to direstory with audio files

    Returns:
        str: file path to a random audio file.
    """
    while True:
        files = [f for f in os.listdir(dir_path) if f.endswith(".npy")]
        if files:
            return os.path.join(dir_path, random.choice(files))
        subdirectories = [d for d in os.listdir(dir_path)
                          if os.path.isdir(os.path.join(dir_path, d))]
        if not subdirectories:
            raise FileNotFoundError("No `.npy` files found in the directory or its subdirectories.")
        dir_path = os.path.join(dir_path, random.choice(subdirectories))

def process(file_path: str = "", directory: str = "", number_of_samples: int = 1):
    """
    Process function that processes the audio file, generates a spectrogram, and optionally
    cleans the data.

    Args:
        sound_path (str): Path to the audio file.
        directory (str): Path to the directory with audio files to generate randomly.
        number_of_samples (int): Number of samples to generate.
        output_path (str): Optional output path for the spectrogram image.
        show (bool): Flag to show the spectrogram using matplotlib.
        mel (bool): Flag to generate a mel-frequency spectrogram.
        clean_data (bool): Flag to clean and normalize the audio data.
        show_axis (bool): Flag to show axis on the spectrogram plot.
    """
    for _ in range(number_of_samples):
        new_npy_path = file_path
        if directory:
            new_npy_path = get_random_npy_path(directory)

        new_npy_path = file_path
        if directory:
            new_npy_path = get_random_npy_path(directory)

        data = np.load(new_npy_path)
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.imshow(data, origin='lower')
        plt.show()


def main(argv: list[str]) -> None:
    """
    Main function that parses command line arguments and runs the processing.

    Args:
        argv (List[str]): List of command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generates a spectrogram from an npy file.")
    parser.add_argument("--file_path", '-f', type=str, help="Path to the .npy file.")
    parser.add_argument("--directory", '-d', type=str,
                        help="Path to the directory containing .npy files.")
    parser.add_argument("--number", '-n', type=int, default=1,
                        help="Number of samples (default is 1).")

    args = parser.parse_args(argv)

    if args.file_path is None and args.directory is None:
        raise ValueError("Please provide either a file path or a directory path.")
    if args.file_path and args.directory:
        raise ValueError("Please provide either a file path or a directory path, not both.")
    if args.file_path and not os.path.exists(args.file_path):
        raise FileNotFoundError(f"The specified file does not exist: {args.file_path}")
    if args.directory and not os.path.isdir(args.directory):
        raise FileNotFoundError(f"The specified directory does not exist: {args.directory}")
    if args.number < 1:
        raise ValueError("The number of samples must be at least 1.")

    process(args.file_path, args.directory, args.number)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
