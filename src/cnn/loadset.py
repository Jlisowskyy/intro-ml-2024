"""
Author: Tomasz Mycielski, 2024

Data loading module
"""
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
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
    le: LabelEncoder # I really wish that LabelEncoder noted that it was deterministic

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

    def __getitem__(self, index: int) -> tuple[Tensor, any]:
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
            f'{self.annotations["file_name"][index][:-4]}_{self.annotations["index"][index]:0>3}.npy'  # pylint: disable=line-too-long
        )
        spectrogram = np.load(spectrogram_path)
        tens = torch.from_numpy(spectrogram).type(torch.float32)
        tens = torch.rot90(tens, dims=(0, 2))
        class_id = self.annotations['classID'][index]
        return tens, class_id

class MultiLabelDataset(Dataset):
    """
    Class featuring methods for retrieving audio data from a multiclass dataset
    """

    # ------------------------------
    # Class fields
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
        self.classes = annotations_file['classID'].unique()
        self.le = LabelEncoder()
        self.le.fit(self.classes)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[Tensor, any]:
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
            self.annotations['file_name'][index][:-4] + '.npy'
        )
        spectrogram = np.load(spectrogram_path)
        tens = torch.from_numpy(spectrogram).type(torch.float32)
        # NOTE rot90 can stay, but I can drop it and we'll have all future models account for it
        # your choice
        tens = torch.rot90(tens, dims=(0, 2))
        class_id = self.le.transform([self.annotations['classID'][index]])[0]
        return tens, class_id

    def get_labels(self) -> list[object]:
        """
        label list getter
        """
        return self.le.classes_

    def get_encoder(self) -> LabelEncoder:
        """
        LabelEncoder getter
        """
        return self.le
