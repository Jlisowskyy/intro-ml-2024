"""
Author: Jakub Lisowski, 2024

File contains manual tests for cutting the wav file
"""

from pathlib import Path

from src.audio.wav import cut_file_to_plain_chunk_files, WavIteratorType

TEST_DATA_PATH = str(Path.resolve(Path(f'{__file__}/../test_data/cut_test_data.wav')))
TEST_OUT_DIR = str(Path.resolve(Path(f'{__file__}/../test_tmp')))

def manual_cut_test() -> None:
    """
    Manual test for cutting the wav file
    """

    cut_file_to_plain_chunk_files(
        TEST_DATA_PATH,
        TEST_OUT_DIR,
        60,
        WavIteratorType.PLAIN)


def manual_test() -> None:
    """
    Manual test for the whole module
    """

    manual_cut_test()
