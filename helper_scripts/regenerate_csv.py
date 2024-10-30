"""
Author: Tomasz Mycielski

Annotations.csv broke? No Problem!
Simply run this script on whichever processed dataset you want to generate a relevant script
"""
import re
from os import walk


PATH = './datasets/daps_split_spectro'

CLASSES = {
    'm1': 0,
    'm2': 0,
    'm3': 1,
    'm4': 0,
    'm5': 0,
    'm6': 1,
    'm7': 0,
    'm8': 1,
    'm9': 0,
    'm10': 0,
    'f1': 1,
    'f2': 0,
    'f3': 0,
    'f4': 0,
    'f5': 0,
    'f6': 0,
    'f7': 1,
    'f8': 1,
    'f9': 0,
    'f10': 0
}

with open('annotations.csv', 'w', encoding='UTF-8') as f:
    f.write('speaker,folder,file_name,index,classID\n')
    for root, _, samples in walk(PATH):
        dirs = root.split('/')
        file = dirs[-1]
        if not file.endswith('.wav'):
            continue
        location = dirs[-2]
        speaker = re.search(r'[fm]\d\d?', file)[0]
        classID = CLASSES[speaker]
        for i in range(len(samples)):
            f.write(f'{speaker},{location},{file},{i},{classID}\n')
