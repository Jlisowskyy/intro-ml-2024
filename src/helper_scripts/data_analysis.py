# pylint: disable=invalid-name
"""
Author: Tomasz Mycielski

Helper script for checking wav data
"""
from os import walk
from os.path import join

from scipy.io import wavfile

from src.constants import DATABASE_PATH, DATABASE_VALID_WAV_SR


def main() -> None:
    """
    Script entry point
    """

    sr_anomaly = False
    min_len = float('inf')
    max_len = 0

    for root, _, files in walk(DATABASE_PATH):
        for i in files:
            if not i.endswith('.wav'):
                continue
            sr, data = wavfile.read(join(root, i))
            duration = len(data) / sr
            print(f'{i}: {sr} Hz, {duration} s')
            if sr != DATABASE_VALID_WAV_SR:
                sr_anomaly = True
            min_len = min(min_len, duration)
            max_len = max(max_len, duration)

    if sr_anomaly:
        print(f'Not every file is {DATABASE_VALID_WAV_SR} Hz')
    else:
        print(f'Every file is {DATABASE_VALID_WAV_SR} Hz')

    print(f'Min duration: {min_len} s')
    print(f'Max duration: {max_len} s')
