"""
Author: Jakub Pietrzak, 2024

Modul for generating spectrograms and showing/saving it
"""

import argparse

import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.pipeline import Pipeline

from src.audio.audio_data import AudioData
from src.audio.spectrogram import gen_mel_spectrogram, gen_spectrogram, save_spectrogram
from src.pipelines.audio_cleaner import AudioCleaner


def process(sound_path: str, output_path: str = None, show: bool = False,
            mel: bool = False, clean_data: bool = False, show_axis: bool = False):
    """
    Process function that processes the audio file, generates a spectrogram, and optionally
    cleans the data.

    Args:
        sound_path (str): Path to the audio file.
        output_path (str): Optional output path for the spectrogram image.
        show (bool): Flag to show the spectrogram using matplotlib.
        mel (bool): Flag to generate a mel-frequency spectrogram.
        clean_data (bool): Flag to clean and normalize the audio data.
        show_axis (bool): Flag to show axis on the spectrogram plot.
    """
    data, samplerate = sf.read(sound_path)
    audio_data = AudioData(data, samplerate)

    if clean_data:
        transformation_pipeline = Pipeline(steps=[
            ('AudioCleaner', AudioCleaner())
        ])
        transformation_pipeline.fit([audio_data])
        audio_data = transformation_pipeline.transform([audio_data])[0]

    if mel:
        spectrogram = gen_mel_spectrogram(audio_data.audio_signal,
                                          audio_data.sample_rate, show_axis)
    else:
        spectrogram = gen_spectrogram(audio_data.audio_signal, audio_data.sample_rate, show_axis)

    if output_path:
        save_spectrogram(spectrogram, output_path)

    if show:
        plt.imshow(spectrogram)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()


def main(argv: list[str]) -> None:
    """
    Main function that parses command line arguments and runs the processing.

    Args:
        argv (List[str]): List of command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generates a spectrogram from an audio file.")
    parser.add_argument("sound_path", type=str, help="Path to the audio file.")
    parser.add_argument('--output', '-o', type=str,
                        help='Optional output file path for the spectrogram.')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Show the spectrogram after generation.')
    parser.add_argument('--clean', '-c', action='store_true',
                        help='Clean and normalize the audio data.')
    parser.add_argument('--mel', '-m', action='store_true',
                        help='Generate a mel-frequency spectrogram.')
    parser.add_argument('--show_axis', '-a', action='store_true',
                        help='Show axis on the spectrogram plot.')

    args = parser.parse_args(argv)
    process(args.sound_path, args.output, args.show, args.mel, args.clean, args.show_axis)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
