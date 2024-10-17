# pylint: disable=invalid-name
"""
Helper script for generating DAPS dataset annotations
"""
from os import walk
from os.path import join

from scipy.io import wavfile

if __name__ == "__main__":
    sr_anomaly = False
    min_len = float('inf')
    max_len = 0

    for root, dirs, files in walk('./datasets/daps'):
        folder = root.rsplit('/')[-1]
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
