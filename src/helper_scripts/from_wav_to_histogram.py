"""
Autor: Jakub Pietrzak, 2024

Modul for generating histograms from audio files
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from src.audio.audio_data import AudioData
from src.audio.spectrogram import gen_mel_spectrogram, save_spectrogram
from src.pipelines.audio_cleaner import AudioCleaner


def cut_audio_to_5_seconds(directory):
    """
    Function that cuts audio files to 5 seconds parts
    Args:
        directory (str): Path to the directory with audio files.
    """
    if not os.path.exists(directory):
        print("directory does not exist")
        return ""

    output_directory = os.path.join(directory, "_5_seconds_audio")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(directory)
    for index,file in enumerate(files):
        if file.endswith(".wav"):
            data, samplerate = sf.read(os.path.join(directory, file))

            #cut into 5 seconds parts
            length=len(data)
            for i in range(0,length,samplerate*5):
                name = os.path.join(output_directory, file[:-4] + "_" + str(i) + ".wav")
                data_part=data[i:i+samplerate*5]
                if len(data_part)<samplerate*5:
                    break
                sf.write(name,  data_part, samplerate)

        print(f"Cutting: Done with {file}, {index}")

    print("Done with cutting audio files to 5 seconds parts")
    return output_directory


def create_spectrograms(directory, denoise=False):
    """
    Function that creates spectrograms from audio files
    Args:
        directory (str): Path to the directory with audio files.
    """
    output_directory = os.path.join(directory, "_spectrograms")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(directory)
    for index,file in enumerate(files):
        if file.endswith(".wav"):
            data, samplerate = sf.read(os.path.join(directory, file))
            audio_data = AudioData(data, samplerate)
            if denoise:
                transformation_pipeline = Pipeline(steps=[
                    ('AudioCleaner', AudioCleaner())
                ])
                transformation_pipeline.fit([data])
                audio_data = transformation_pipeline.transform([audio_data])[0]
            spectrogram = gen_mel_spectrogram(audio_data.audio_signal, samplerate)
            save_spectrogram(spectrogram, os.path.join(output_directory, file[:-4] + ".png"))
            print(f"Spectrogram: Done with {file}, {index}")
    print("Done with creating spectrograms")
    return output_directory


def create_rgb_histogram(directory):  # pylint: disable=too-many-locals
    """
    Function that creates rgb histograms from spectrograms
    Args:
        directory (str): Path to the directory with spectrograms.
    """
    files = os.listdir(directory)
    class1 = ["f1", "f7", "f8", "m3", "m6", "m8"]
    array_class1 = [[0] * 256 for _ in range(3)]
    array_class0 = [[0] * 256 for _ in range(3)]

    num_class1 = 0
    num_class0 = 0

    for index, file_name in enumerate(files):
        image = Image.open(os.path.join(directory, file_name))
        image_array = np.array(image)

        # Separate the color channels
        r_channel = image_array[:, :, 0].flatten()
        g_channel = image_array[:, :, 1].flatten()
        b_channel = image_array[:, :, 2].flatten()

        counts_r = np.bincount(r_channel, minlength=256)
        counts_g = np.bincount(g_channel, minlength=256)
        counts_b = np.bincount(b_channel, minlength=256)

        if file_name.split("_")[0] in class1:
            num_class1 += 1
            for i in range(256):
                array_class1[0][i] += counts_r[i]
                array_class1[1][i] += counts_g[i]
                array_class1[2][i] += counts_b[i]
        else:
            num_class0 += 1
            for i in range(256):
                array_class0[0][i] += counts_r[i]
                array_class0[1][i] += counts_g[i]
                array_class0[2][i] += counts_b[i]
        print(f"Histogram: Done with {file_name}, {index}")
    print("Done with creating rgb histograms")

    for i in range(0, 255):
        for j in range(3):
            if num_class1!=0:
                array_class1[j][i] /= num_class1
            if num_class0!=0:
                array_class0[j][i] /= num_class0

    return array_class1, array_class0

def show_rgb_histogram(array_class1, array_class0):
    """
    shows rgb histogram

    Args:
        array_class1 (_type_): array [3][256]
        array_class0 (_type_): array [3][256]
    """
    intensity_levels = np.arange(256)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(intensity_levels, array_class0[0], color='red', alpha=0.5, label='Red')
    plt.bar(intensity_levels, array_class0[1], color='green', alpha=0.5, label='Green')
    plt.bar(intensity_levels, array_class0[2], color='blue', alpha=0.5, label='Blue')
    plt.title('RGB Histogram - Class 0')
    plt.xlabel('Intensity Level')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
    plt.legend()
    plt.grid(axis='y')

    plt.subplot(1, 2, 2)
    plt.bar(intensity_levels, array_class1[0], color='red', alpha=0.5, label='Red')
    plt.bar(intensity_levels, array_class1[1], color='green', alpha=0.5, label='Green')
    plt.bar(intensity_levels, array_class1[2], color='blue', alpha=0.5, label='Blue')
    plt.title('RGB Histogram - Class 1')
    plt.xlabel('Intensity Level')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
    plt.legend()
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

def delete_silence(data, threshold):
    """
    Function that deletes silence from audio data
    Args:
        data (np.array): Audio data.
        threshold (float): Threshold for silence.
    """
    return data[np.where(np.abs(data) > threshold)]


def delete_silence_in_files(directory, threshold):
    """
    Function that deletes silence from audio data
    Args:
        data (np.array): Audio data.
        threshold (float): Threshold for silence.
    """
    output_directory = os.path.join(directory, "_silence_deleted")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    files = os.listdir(directory)
    for index,file in enumerate(files):
        if file.endswith(".wav"):
            data, samplerate = sf.read(os.path.join(directory, file))
            data = delete_silence(data, threshold)
            sf.write(os.path.join(output_directory, file), data, samplerate)
            print(f"Silence: Done with {file}, {index}")
    print("Done with deleting silence")
    return output_directory


def process_directory(directory: str) -> None:
    """
    Main function that processes the audio files, generates spectrograms, and optionally
    creates rgb histograms.

    :param directory: Path to the directory with audio files
    """

    output_directory = delete_silence_in_files(directory, 0.01)
    output_directory = cut_audio_to_5_seconds(output_directory)
    output_directory = create_spectrograms(output_directory, True)
    array_class1, array_class0 = create_rgb_histogram(output_directory)
    show_rgb_histogram(array_class1, array_class0)


DEFAULT_DIR = str(Path.resolve(Path(f'{__file__}/../work_dir')))


def main(args: list[str]) -> None:
    """
    Program entry point

    :param args: list of arguments
    """

    if len(args) == 0:
        directory = DEFAULT_DIR
    elif len(args) == 1:
        directory = args[0]
    else:
        raise ValueError("Invalid number of arguments")
    process_directory(directory)


if __name__ == "__main__":
    main(sys.argv[1:])
