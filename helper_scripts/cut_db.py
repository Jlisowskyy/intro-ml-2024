"""
Author: Jakub Lisowski, 2024

Simple script cutting the database into smaller parts using simple fixed length window
"""

import os

from src.audio.wav import cut_file_to_plain_chunk_files, WavIteratorType

if __name__ == "__main__":
    DATASET_PATH = './datasets/daps'
    DESTINATION_DIR = './datasets/cut_chunks'
    WINDOW_LENGTH_SECONDS = 5.0
    ITERATOR_TYPE = WavIteratorType.PLAIN

    os.makedirs(DESTINATION_DIR, exist_ok=True)

    for root, dirs, files in os.walk(DATASET_PATH):
        folder = root.rsplit('/')[-1]
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_name}")

                cut_file_to_plain_chunk_files(file_path, DESTINATION_DIR, WINDOW_LENGTH_SECONDS,
                                              ITERATOR_TYPE)

    print("All files have been processed and cut into chunks.")
