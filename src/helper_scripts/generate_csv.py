"""
Author: Tomasz Mycielski

Helper script for generating DAPS dataset annotations
"""

import re
from os import walk

from src.constants import DATABASE_PATH, DATABASE_ANNOTATIONS_PATH


def main() -> None:
    with open(DATABASE_ANNOTATIONS_PATH, 'w', encoding='UTF-8') as f:
        f.write('file_name,speaker,folder,classID\n')

        # pylint: disable=invalid-name
        class_id: int = 0

        for root, dirs, files in walk(DATABASE_PATH):
            print(root, dirs, files)
            folder = root.rsplit('/')[-1]
            for i in files:
                if not i.endswith('.wav'):
                    continue
                if re.match('^(m[368])|(f[178][^0])', i):
                    class_id = 1
                else:
                    class_id = 0
                f.write(f'{i},{i[0:2] if i[1:3] != "10" else i[0:3]},{folder},{class_id}\n')


if __name__ == "__main__":
    main()
