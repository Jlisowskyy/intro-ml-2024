"""
Author: Michał Kwiatkowski

Helper script for generating files with noises
"""

from os import listdir, path

from src.constants import DATABASE_NOISES, DATABASE_OUT_NOISES, MODEL_WINDOW_LENGTH, WavIteratorType 

from src.pipeline.wav import cut_file_to_plain_chunk_files


def main() -> None:

    for file in listdir(DATABASE_NOISES):
        file_path = path.join(DATABASE_NOISES, file)

        # Omit nonaudio files
        if (not file.endswith('.wav')
            or file.startswith('.')):
            continue

        cut_file_to_plain_chunk_files(file_path = file_path,
                                      destination_dir = DATABASE_OUT_NOISES,
                                      window_length_seconds = MODEL_WINDOW_LENGTH,
                                      iterator_type = WavIteratorType.OVERLAPPING)
        