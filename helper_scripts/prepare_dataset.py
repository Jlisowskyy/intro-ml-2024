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
from src.pipelines.spectrogram_generator import gen_mel_spectrogram


def right_pad_if_necessary(audio: np.ndarray, sample_count: int) -> np.ndarray:
    """
    Pad audio with zeros if too short

    Parameters
    ----------
    audio: :class:`np.ndarray`
        audio data
    sample_count: `int`
        expected amount of frames

    Returns
    -------
    :class:`np.ndarray`
        padded or unchanged audio data
    """
    if len(audio) < sample_count:
        audio = np.pad(audio, (0, sample_count - len(audio)))
    return audio


WAV_ITERATOR_TYPE = wav.WavIteratorType.PLAIN
WINDOW_LENGTH = 5 # affects CNN

with open('annotations.csv', 'w', encoding='UTF-8') as f:
    f.write('speaker,folder,file_name,index,classID\n')
    for root, dirs, files in walk('./datasets/daps/'):
        folder = root.rsplit('/')[-1]
        newroot = root.replace('daps', 'daps_split_spectro')
        for file in tqdm(files, colour='magenta'):
            if not file.endswith('.wav'):
                continue
            if re.match('^(m[368])|(f[178][^0])', file):
                CLASSID = 1
            else:
                CLASSID = 0
            it = wav.load_wav_with_window(path.join(root, file), WINDOW_LENGTH, 0)
            sr = it.get_frame_rate()
            COUNTER = 0
            for audio_data in it:
                audio_data = AudioData.to_float(audio_data)

                if not detect_speech.is_speech(audio_data, sr):
                    continue

                audio_data = denoise.denoise(audio_data, sr)
                audio_data = normalize.normalize(audio_data, sr,
                                                 normalize.NormalizationType.MEAN_VARIANCE)
                audio_data = right_pad_if_necessary(audio_data, WINDOW_LENGTH * sr)
                spectrogram = gen_mel_spectrogram(audio_data, 16000,
                                                  width=300, height=400)
                if not path.exists(path.join(newroot, file)):
                    makedirs(path.join(newroot, file))
                np.save(path.join(newroot, file, f'{file[:-4]}_{COUNTER:0>3}.npy'), spectrogram)
                f.write(
                    f'{file[0:2] if file[1:3] != "10" else file[0:3]},{folder},{file},{COUNTER},{CLASSID}\n')  # pylint: disable=line-too-long
                COUNTER += 1
