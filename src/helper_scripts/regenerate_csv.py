"""
Author: Tomasz Mycielski

Annotations.csv broke? No Problem!
Simply run this script on whichever processed dataset you want to generate a relevant script
"""
import re
from os import walk
from src.constants import SPEAKER_CLASSES

PATH = './datasets/daps_split_spectro'


with open('annotations.csv', 'w', encoding='UTF-8') as f:
    f.write('speaker,folder,file_name,index,classID\n')
    for root, _, samples in walk(PATH):
        dirs = root.split('/')
        file = dirs[-1]
        if not file.endswith('.wav'):
            continue
        location = dirs[-2]
        speaker = re.search(r'[fm]\d\d?', file)[0]
        classID = SPEAKER_CLASSES[speaker]
        for i in range(len(samples)):
            f.write(f'{speaker},{location},{file},{i},{classID}\n')
