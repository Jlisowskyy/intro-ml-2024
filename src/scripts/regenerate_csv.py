"""
Author: Tomasz Mycielski

Annotations.csv broke? No Problem!
Simply run this script on whichever processed dataset you want to generate a relevant script
"""

from os import walk

from src.constants import DATABASE_ANNOTATIONS_PATH, DATABASE_OUT_PATH, CLASSES


def main() -> None:
    """
    Script entry point
    """

    with open(DATABASE_ANNOTATIONS_PATH, 'w', encoding='UTF-8') as f:
        f.write('folder,file_name,classID\n')
        for root, _, files in walk(DATABASE_OUT_PATH):
            class_id = root.split('/')[-1]
            if class_id not in CLASSES:
                class_id = 'unknown'
            for i in files:
                f.write(f'{root},{i},{class_id}\n')
