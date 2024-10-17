"""
Helper script for generating DAPS dataset annotations
"""
from os import walk
import re

with open('annotations.csv', 'w', encoding='UTF-8') as f:
    f.write('file_name,speaker,folder,classID\n')
    for root, dirs, files in walk('./datasets/daps'):
        print(root, dirs, files)
        folder = root.rsplit('/')[-1]
        for i in files:
            if re.match('^(m[368])|(f[178][^0])', i):
                CLASSID = 1
            else:
                CLASSID = 0
            f.write(f'{i},{i[0:2] if i[1:3] != "10" else i[0:3]},{folder},{CLASSID}\n')
