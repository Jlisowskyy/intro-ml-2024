"""
Data loading module
"""
from os.path import join
from torch.utils.data import Dataset
from torch import nn, mean  # pylint: disable=no-name-in-module
import torchaudio
import pandas as pd


class DAPSDataset(Dataset):
    """
    Class featuring methods for retrieving audio data from the DAPS dataset
    """
    def __init__(
            self,
            annotations_file: str,
            root: str,
            transformation: nn.Module,
            sample_rate: int,
            sample_count: int,
            device: str = 'cpu'):
        """
        Parameters
        ----------
        annotations_file: :class:`str` or a pathlike
            location of a csv file describing the data

        root: :class:`str` or a pathlike
            location of the dataset
        
        transformation: :class:`torch.nn.Module`
            set of transformations to apply on the data

        sample_rate: :class:`int`
            target audio sample rate
        
        sample_count: :class:`int`
            sample rate * seconds to load
        
        device: :class:`str`
            can be 'cuda' or 'cpu', device to load data onto
        """
        self.annotations = pd.read_csv(annotations_file)
        self.root = root
        self.transformation = transformation.to(device)
        self.sample_rate = sample_rate
        self.sample_count = sample_count
        self.device = device
        self.offset_seconds = 20

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Retrieve item from the dataset, random access

        Parameters
        ----------
        index: :class:`int`
            access location
        """
        audio_path = join(self.root, self.annotations['folder'][index],
                          self.annotations['file_name'][index])
        class_id = self.annotations['classID'][index]
        signal, sr = torchaudio.load(audio_path, frame_offset=44100 * self.offset_seconds)
        signal = signal.to(self.device)
        resample = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
        signal = resample(signal)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, class_id

    def _right_pad_if_necessary(self, signal):
        # pad signal if too short
        length_signal = signal.shape[1]
        if length_signal < self.sample_count:
            num_missing_samples = self.sample_count - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = nn.functional.pad(signal, last_dim_padding)
        return signal

    def _cut_if_necessary(self, signal):
        # trim signal if too long
        if signal.shape[1] > self.sample_count:
            signal = signal[:, :self.sample_count]
        return signal

    def _mix_down_if_necessary(self, signal):
        # make mono if stereo
        if signal.shape[0] > 1:
            signal = mean(signal, dim=0, keepdim=True)
        return signal
