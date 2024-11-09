"""
Author: Tomasz Mycielski

Annotations.csv broke? No Problem!
Simply run this script on whichever processed dataset you want to generate a relevant script
"""
import re
from os import walk

from src.constants import SPEAKER_CLASSES, DATABASE_ANNOTATIONS_PATH, DATABASE_OUT_PATH


def main() -> None:
    """
    Script entry point
    """

    with open(DATABASE_ANNOTATIONS_PATH, 'w', encoding='UTF-8') as f:
        f.write('speaker,folder,file_name,index,classID\n')
        for root, _, samples in walk(DATABASE_OUT_PATH):
            dirs = root.split('/')
            file = dirs[-1]
            if not file.endswith('.wav'):
                continue
            location = dirs[-2]
            speaker = re.search(r'[fm]\d\d?', file)[0]
            classID = SPEAKER_CLASSES[speaker]
            for i in range(len(samples)):
                f.write(f'{speaker},{location},{file},{i},{classID}\n')


if __name__ == "__main__":
    main()
