"""
Author: Jakub Lisowski, 2024

Simple script cutting the database into smaller parts using simple fixed length window
"""

import os
from src.audio.wav import cut_file_to_plain_chunk_files, WavIteratorType

if __name__ == "__main__":
    dataset_path = './datasets/daps'
    destination_dir = './datasets/cut_chunks'
    window_length_seconds = 5.0
    iterator_type = WavIteratorType.PLAIN

    os.makedirs(destination_dir, exist_ok=True)

    for root, dirs, files in os.walk(dataset_path):
        folder = root.rsplit('/')[-1]
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_name}")

                cut_file_to_plain_chunk_files(file_path, destination_dir, window_length_seconds, iterator_type)

    print("All files have been processed and cut into chunks.")
