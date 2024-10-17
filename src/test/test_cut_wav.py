"""
Author: Jakub Lisowski, 2024

File contains manual tests for cutting the wav file
"""

from src.audio.wav import cut_file_to_plain_chunk_files, WavIteratorType



def manual_cut_test() -> None:
    """
    Manual test for cutting the wav file
    """

    cut_file_to_plain_chunk_files(
        "data/voice.wav",
        "data/voice_cut",
        5,
        WavIteratorType.PLAIN)


def manual_test() -> None:
    """
    Manual test for the whole module
    """

    manual_cut_test()
