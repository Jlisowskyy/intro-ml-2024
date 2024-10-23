"""
Author: Tomasz Mycielski, 2024

Data loading module
"""
from os.path import join

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset


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
    device: str

    # ------------------------------
    # Class implementation
    # ------------------------------

    def __init__(
            self,
            annotations_file: str,
            root: str,
            device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        annotations_file: :class:`str` or a pathlike
            location of a csv file describing the data

        root: :class:`str` or a pathlike
            location of the dataset
        
        device: :class:`str`
            can be 'cuda' or 'cpu', device to load data onto
        """
        self.annotations = pd.read_csv(annotations_file)
        self.root = root
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
        spectrogram_path = join(
            self.root,
            self.annotations['folder'][index],
            self.annotations['file_name'][index],
            f'{self.annotations["file_name"][index][:-4]}_{self.annotations["index"][index]:0>3}.npy'
        )
        spectrogram = np.load(spectrogram_path)
        tens = torch.from_numpy(spectrogram).type(torch.float32)
        tens = torch.rot90(tens, dims=(0, 2))
        class_id = self.annotations['classID'][index]
        return (tens, class_id)

if __name__ == '__main__':
    d = DAPSDataset('annotations.csv', './datasets/daps_split_spectro/', 'cuda')
    print(d[0])
