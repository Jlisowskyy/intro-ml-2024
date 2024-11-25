"""
Author: Tomasz Mycielski

Helper script for generating a dataset and a relevant annotations file
"""

from os import walk, path, makedirs

import numpy as np
from tqdm import tqdm

from src.constants import MODEL_WINDOW_LENGTH, DATABASE_PATH, \
    DATABASE_OUT_NAME, DATABASE_CUT_ITERATOR, CLASSES, \
    DATABASE_ANNOTATIONS_PATH, NORMALIZATION_TYPE, DATABASE_NAME

from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.wav import FlattenWavIterator, AudioDataIterator


def main(dry: bool = False) -> None:
    """
    Script entry point
    """

    # TODO: add NUL support for the windows users
    db_path = DATABASE_ANNOTATIONS_PATH if not dry else '/dev/null'
    with open(db_path, 'w', encoding='UTF-8') as f:
        f.write('folder,file_name,classID\n')

        for root, _, files in walk(DATABASE_PATH):
            folder = root.rsplit('/')[-1]
            new_root = root.replace(DATABASE_NAME, DATABASE_OUT_NAME)

            for file in tqdm(files, colour='magenta'):
                # Omit annoying hidden mac files
                if (not file.endswith('.wav')
                    or file.startswith('.')
                    or folder == '_background_noise_'):
                    continue
                it = FlattenWavIterator(path.join(root, file), MODEL_WINDOW_LENGTH,
                                        DATABASE_CUT_ITERATOR)

                sr = it.get_first_iter().get_frame_rate()
                it = AudioDataIterator(it)


                for audio_data in it:
                    # Pad not full files
                    if len(audio_data.audio_signal) < MODEL_WINDOW_LENGTH * sr:
                        audio_data.audio_signal = np.pad(audio_data.audio_signal,
                               (0,MODEL_WINDOW_LENGTH * sr - len(audio_data.audio_signal)),
                               constant_values=(0,0))

                    spectrogram = process_audio(audio_data, NORMALIZATION_TYPE)

                    class_id = folder if folder in CLASSES else 'unknown'
                    data = f'{new_root},{file},{class_id}'

                    if dry:
                        print(data)
                    else:
                        if not path.exists(new_root):
                            makedirs(path.join(new_root))
                        np.save(path.join(new_root, f'{file[:-4]}.npy'), spectrogram)
                        f.write(data + '\n')
                    break # Apologies to future me for the legacy code, it was easier to adapt it
