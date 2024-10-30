"""
Author: Tomasz Mycielski

Helper script for generating a dataset and a relevant annotations file
while splitting the files into smaller ones
"""
import re
from os import walk, path, makedirs

import numpy as np
from tqdm import tqdm

import src.constants
from src.audio import normalize, denoise, detect_speech
from src.audio.audio_data import AudioData
from src.audio.spectrogram import gen_mel_spectrogram
from src.audio.wav import FlattenWavIterator
from src.constants import MODEL_WINDOW_LENGTH, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH, \
    DATABASE_PATH, DATABASE_OUT_NAME, DATABASE_CUT_ITERATOR, SPEAKER_CLASSES

with open('annotations.csv', 'w', encoding='UTF-8') as f:
    f.write('speaker,folder,file_name,index,classID\n')

    for root, dirs, files in walk(DATABASE_PATH):
        folder = root.rsplit('/')[-1]
        new_root = root.replace('daps', DATABASE_OUT_NAME)

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

            sub_file_counter: int = 0
            for audio_data in it:
                audio_data = AudioData.to_float(audio_data)

                # Omit not full chunks to avoid filling the dataset with silence
                if len(audio_data) < MODEL_WINDOW_LENGTH * sr:
                    continue

                if not detect_speech.is_speech(audio_data, int(sr)):
                    continue

                audio_data = denoise.denoise(audio_data, sr)
                audio_data = normalize.normalize(audio_data, sr,
                                                 src.constants.NormalizationType.MEAN_VARIANCE)
                spectrogram = gen_mel_spectrogram(audio_data, int(sr),
                                                  width=SPECTROGRAM_WIDTH,
                                                  height=SPECTROGRAM_HEIGHT)

                if not path.exists(path.join(new_root, file)): # TODO: add [:-4]
                    makedirs(path.join(new_root, file))
                np.save(path.join(new_root, file, f'{file[:-4]}_{sub_file_counter:0>3}.npy'),
                        spectrogram)
                f.write(
                    f'{speaker},{folder},{file},{sub_file_counter},{data_class_id}\n')  # pylint: disable=line-too-long
                sub_file_counter += 1
