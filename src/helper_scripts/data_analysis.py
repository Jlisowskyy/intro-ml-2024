# pylint: disable=invalid-name
"""
Author: Tomasz Mycielski

Helper script for checking wav data
"""
from os import walk
from os.path import join

from scipy.io import wavfile


def main() -> None:
    """
    Script entry point
    """

    sr_anomaly = False
    min_len = float('inf')
    max_len = 0

    for root, _, files in walk('./datasets/daps'):
        for i in files:
            if not i.endswith('.wav'):
                continue
            sr, data = wavfile.read(join(root, i))
            duration = len(data) / sr
            print(f'{i}: {sr} Hz, {duration} s')
            if sr != 44100:
                sr_anomaly = True
            min_len = min(min_len, duration)
            max_len = max(max_len, duration)

    if sr_anomaly:
        print('Not every file is 44.1 kHz')
    else:
        print('Every file is 44.1 kHz')

    print(f'Min duration: {min_len} s')
    print(f'Max duration: {max_len} s')


if __name__ == "__main__":
    main()
