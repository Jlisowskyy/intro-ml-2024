"""
Author: Tomasz Mycielski

Helper script for generating a dataset and a relevant annotations file
while splitting the files into smaller ones
"""
import re
from os import walk, path, makedirs

import numpy as np
from tqdm import tqdm

from src.audio import wav, normalize, denoise, detect_speech
from src.audio.audio_data import AudioData
from src.audio.spectrogram import gen_mel_spectrogram
from src.constants import MODEL_WINDOW_LENGTH, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH

def dump_database(file_path: str) -> None:
    """
    Dumps the database which resides in the given file path.
    """
    pass

with open('annotations.csv', 'w', encoding='UTF-8') as f:
    f.write('speaker,folder,file_name,index,classID\n')

    for root, dirs, files in walk('./datasets/daps/'):
        folder = root.rsplit('/')[-1]
        new_root = root.replace('daps', 'daps_split_spectro')

        for file in tqdm(files, colour='magenta'):
            # Omit annoying hidden mac files
            if not file.endswith('.wav') or file.startswith('.'):
                continue

            if re.match('^(m[368])|(f[178][^0])', file):
                data_class_id = 1
            else:
                data_class_id = 0

            it = wav.load_wav_with_window(path.join(root, file), MODEL_WINDOW_LENGTH, 0)
            sr = it.get_frame_rate()

            sub_file_counter = 0
            for audio_data in it:
                # TODO: PLACE PIPLINE HERR!!!!


                audio_data = AudioData.to_float(audio_data)

                if len(audio_data) < MODEL_WINDOW_LENGTH * sr:
                    continue

                if not detect_speech.is_speech(audio_data, int(sr)):
                    continue


                audio_data = denoise.denoise(audio_data, sr)
                audio_data = normalize.normalize(audio_data, sr,
                                                 normalize.NormalizationType.MEAN_VARIANCE)
                spectrogram = gen_mel_spectrogram(audio_data, int(sr),
                                                  width=SPECTROGRAM_HEIGHT,
                                                  height=SPECTROGRAM_WIDTH)

                # TODO: END OF PIPELINE

                if not path.exists(path.join(new_root, file)):
                    makedirs(path.join(new_root, file))
                np.save(path.join(new_root, file, f'{file[:-4]}_{sub_file_counter:0>3}.npy'),
                        spectrogram)
                f.write(
                    f'{file[0:2] if file[1:3] != "10" else file[0:3]},{folder},{file},{sub_file_counter},{data_class_id}\n')  # pylint: disable=line-too-long
                sub_file_counter += 1
