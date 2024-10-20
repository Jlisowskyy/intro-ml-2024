"""
Author: Tomasz Mycielski

Helper script for generating a dataset and a relevant annotations file
while splitting the files into smaller ones
"""
import re
from os import walk, path

from src.audio import wav

WAV_ITERATOR_TYPE = wav.WavIteratorType.PLAIN
WINDOW_LENGTH = 5

with open('annotations.csv', 'w', encoding='UTF-8') as f:
    f.write('speaker,folder,file_name,index,classID\n')
    for root, dirs, files in walk('./datasets/daps'):
        folder = root.rsplit('/')[-1]
        newroot = root.replace('daps', 'daps_split')
        for file in files:
            if not file.endswith('.wav'):
                continue
            if re.match('^(m[368])|(f[178][^0])', file):
                CLASSID = 1
            else:
                CLASSID = 0
            COUNT = wav.cut_file_to_plain_chunk_files(
                path.join(root, file),
                path.join(newroot, file),
                WINDOW_LENGTH,
                WAV_ITERATOR_TYPE
            )
            for i in range(COUNT):
                f.write(f'{file[0:2] if file[1:3] != "10" else file[0:3]},{folder},{file},{i},{CLASSID}\n')  # pylint: disable=line-too-long
