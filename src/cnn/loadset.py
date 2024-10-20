"""
Author: Tomasz Mycielski, 2024

Data loading module
"""
from os.path import join

import numpy as np
import pandas as pd
import librosa
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

from ..pipelines.spectrogram_generator import gen_spectrogram
from ..audio import denoise, normalize


class DAPSDataset(Dataset):
    """
    Class featuring methods for retrieving audio data from the DAPS dataset
    """

    # ------------------------------
    # Class fields
    # ------------------------------

    annotations: pd.DataFrame
    root: str
    transformation: nn.Module
    sample_rate: int
    sample_count: int
    device: str
    offset_seconds: int

    # ------------------------------
    # Class implementation
    # ------------------------------

    def __init__(
            self,
            annotations_file: str,
            root: str,
            target_sample_rate: int,
            sample_count: int,
            device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        annotations_file: :class:`str` or a pathlike
            location of a csv file describing the data

        root: :class:`str` or a pathlike
            location of the dataset

        target_sample_rate: :class:`int`
            target audio sample rate
        
        sample_count: :class:`int`
            sample rate * seconds to load
        
        device: :class:`str`
            can be 'cuda' or 'cpu', device to load data onto
        """
        self.width = 300
        self.height = 400
        self.annotations = pd.read_csv(annotations_file)
        self.root = root
        self.target_sample_rate = target_sample_rate
        self.sample_count = sample_count
        self.device = device

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """
        Retrieve item from the dataset, random access

        Parameters
        ----------
        index: :class:`int`
            access location
        """
        audio_path = join(
            self.root,
            self.annotations['folder'][index],
            self.annotations['file_name'][index],
            f'{self.annotations["file_name"][index][:-4]}_{self.annotations["index"][index]:0>3}.wav'  # pylint: disable=line-too-long
        )
        audio_data, sr = librosa.load(audio_path, sr=self.target_sample_rate)=
        audio_data = denoise.denoise(audio_data, sr)
        audio_data = self._right_pad_if_necessary(audio_data)
        spectrogram = gen_spectrogram(audio_data, self.target_sample_rate,
                                      width=self.width, height=self.height)

        tens = torch.from_numpy(spectrogram).type(torch.float32)
        tens = torch.rot90(tens, dims=(0, 2))
        class_id = self.annotations['classID'][index]
        return (tens, class_id)

    def _right_pad_if_necessary(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad audio with zeros if too short

        Parameters
        ----------
        audio: :class:`np.ndarray`
            audio_data
        """
        if len(audio) < self.sample_count:
            audio = np.pad(audio, (0, self.sample_count - len(audio)))
        return audio


if __name__ == '__main__':
    d = DAPSDataset('annotations.csv', './datasets/daps_split/', 16000, 'cuda')
    print(d[0])
