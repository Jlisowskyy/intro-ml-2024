"""
Author: Jakub Lisowski, 2024

This script is used to display spectrogram from the database.
"""
from os.path import join
from sys import argv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.constants import DATABASE_OUT_PATH, DATABASE_ANNOTATIONS_PATH


def display_db_spectrogram(db_path: str, annotations_path: str, file_names: list[str]) -> None:
    """
    Display spectrogram from the database.

    Parameters
    ----------
    :param file_names: names of files that spectrogram should be displayed
    :param db_path: str, path to the database
    :param annotations_path: str, path to the annotations
    """

    annotations = pd.read_csv(annotations_path)
    for idx in range(0, len(annotations)):
        file_name = annotations['file_name'][idx]

        if file_name not in file_names:
            continue

        spectrogram_path = join(
            db_path,
            annotations['folder'][idx],
            annotations['file_name'][idx],
            f'{annotations["file_name"][idx][:-4]}_{annotations["index"][idx]:0>3}.npy'
        )
        spectrogram = np.load(spectrogram_path)

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'Spectrogram: {file_name}')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


def main(args: list[str]):
    """
    Main function for the script.

    Parameters
    ----------

    :param args: list of strings, names of the files spectrogram to display
    """

    display_db_spectrogram(DATABASE_OUT_PATH, DATABASE_ANNOTATIONS_PATH, args)


if __name__ == '__main__':
    main(argv[1:])
