"""
Author: Michał Kwiatkowski

Helper script for generating files with noises.
This script processes `.wav` files in a source directory, splits them into chunks
of a specified length, and saves the resulting chunks to a destination directory.
"""

from os import listdir, path

from src.constants import DATABASE_NOISES, DATABASE_OUT_NOISES, MODEL_WINDOW_LENGTH, WavIteratorType
from src.pipeline.wav import cut_file_to_plain_chunk_files

def main() -> None:
    """
    Main function for processing `.wav` files in the DATABASE_NOISES directory.
    It splits each file into smaller chunks based on the specified window length 
    and saves the chunks into the DATABASE_OUT_NOISES directory.

    The script skips non-audio files and hidden files (those starting with '.').
    """
    for file in listdir(DATABASE_NOISES):
        file_path = path.join(DATABASE_NOISES, file)

        # Omit non-audio files
        if (not file.endswith('.wav') or file.startswith('.')):
            continue

        # Split the audio file into chunks and save them
        cut_file_to_plain_chunk_files(
            file_path=file_path,
            destination_dir=DATABASE_OUT_NOISES,
            window_length_seconds=MODEL_WINDOW_LENGTH,
            iterator_type=WavIteratorType.OVERLAPPING
        )
