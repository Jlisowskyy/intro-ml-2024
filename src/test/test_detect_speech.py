"""
Author: Jakub Lisowski

File tests quality of silence removal
"""

from pathlib import Path
from typing import Callable

import numpy as np

from src.constants import WavIteratorType
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.wav import load_wav

TEST_FILE_NAME1 = "f5733968_nohash_4.wav"
TEST_FILE_NAME2 = "f6581345_nohash_0.wav"
TEST_FILE_NAME3 = "f2_script1_ipad_office1_35000.wav"

TEST_FILE_PATH1 = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME1}')))
TEST_FILE_OUTPUT_AFTER_PATH1 = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME1}_after_silence_removal.wav')))
TEST_FILE_PATH2 = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME2}')))
TEST_FILE_OUTPUT_AFTER_PATH2 = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME2}_after_silence_removal.wav')))
TEST_FILE_PATH3 = str(
    Path.resolve(Path(f'{__file__}/../test_data/{TEST_FILE_NAME3}')))
TEST_FILE_OUTPUT_AFTER_PATH3 = str(
    Path.resolve(Path(f'{__file__}/../test_tmp/{TEST_FILE_NAME3}_after_silence_removal.wav')))

TEST_FILES = [
    (TEST_FILE_PATH1, TEST_FILE_OUTPUT_AFTER_PATH1),
    (TEST_FILE_PATH2, TEST_FILE_OUTPUT_AFTER_PATH2),
    (TEST_FILE_PATH3, TEST_FILE_OUTPUT_AFTER_PATH3)
]


def silence_removal_test() -> None:
    """
    Run the manual test for the silence removal module.
    """

    for test_file, test_out in TEST_FILES:
        it = load_wav(test_file, 0, WavIteratorType.PLAIN)

        transform_func: Callable[[AudioData], np.ndarray] = lambda \
                x: AudioCleaner.remove_silence_raw(x.audio_signal, x.sample_rate)
        it.transform(transform_func)

        it.set_window_size(it.get_num_frames())

        for _, num in enumerate(it):
            print(num)
