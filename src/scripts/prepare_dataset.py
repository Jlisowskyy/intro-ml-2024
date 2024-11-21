"""
Author: Tomasz Mycielski

Helper script for generating a dataset and a relevant annotations file
while splitting the files into smaller ones
"""
import re
from os import walk, path, makedirs

import numpy as np
from tqdm import tqdm

from src.audio import detect_speech
from src.audio.wav import FlattenWavIterator, AudioDataIterator
from src.constants import MODEL_WINDOW_LENGTH, DATABASE_PATH, \
    DATABASE_OUT_NAME, DATABASE_CUT_ITERATOR, SPEAKER_CLASSES, \
    DATABASE_ANNOTATIONS_PATH, NORMALIZATION_TYPE, DATABASE_NAME
from src.pipeline.base_preprocessing_pipeline import process_audio


def main() -> None:
    """
    Script entry point
    """

    with open(DATABASE_ANNOTATIONS_PATH, 'w', encoding='UTF-8') as f:
        f.write('speaker,folder,file_name,index,classID\n')

        for root, _, files in walk(DATABASE_PATH):
            folder = root.rsplit('/')[-1]
            new_root = root.replace(DATABASE_NAME, DATABASE_OUT_NAME)

            for file in tqdm(files, colour='magenta'):
                # Omit annoying hidden mac files
                if not file.endswith('.wav') or file.startswith('.'):
                    continue

                # RIP /^(m[368])|(f[178][^0])/
                speaker = re.search(r'[fm]\d\d?', file)[0]
                data_class_id = SPEAKER_CLASSES[speaker]
                it = FlattenWavIterator(path.join(root, file), MODEL_WINDOW_LENGTH,
                                        DATABASE_CUT_ITERATOR)
                sr = it.get_first_iter().get_frame_rate()
                it = AudioDataIterator(it)

                sub_file_counter: int = 0
                for audio_data in it:
                    # Omit not full chunks to avoid filling the dataset with silence
                    if len(audio_data.audio_signal) < MODEL_WINDOW_LENGTH * sr:
                        continue

                    if not detect_speech.is_speech(audio_data.audio_signal):
                        continue

                    spectrogram = process_audio(audio_data, NORMALIZATION_TYPE)

                    if not path.exists(path.join(new_root, file)):  # TODO: add [:-4]
                        makedirs(path.join(new_root, file))
                    np.save(path.join(new_root, file, f'{file[:-4]}_{sub_file_counter:0>3}.npy'),
                            spectrogram)
                    f.write(
                        f'{speaker},{folder},{file},{sub_file_counter},{data_class_id}\n')  # pylint: disable=line-too-long
                    sub_file_counter += 1
