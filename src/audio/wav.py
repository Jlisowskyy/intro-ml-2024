# pylint: disable=no-member
"""
Author: Jakub Lisowski, 2024

This module provides a simple iterator over the samples of a WAV file.

"""

import os
import wave
from abc import ABC, abstractmethod
from collections.abc import Generator

import numpy as np

from src.constants import WavIteratorType


class WavIteratorBase(ABC):
    """
    Base class for the WAV file iterator
    """

    # ------------------------------
    # class fields
    # ------------------------------

    #pylint: disable=too-many-instance-attributes
    _file_path: str
    _window_size_frames: int

    _window_index: int
    _channel_index: int

    _frame_rate: int
    _num_frames: int
    _sample_width: int
    _num_channels: int

    _audio_data: np.ndarray

    # ------------------------------
    # class creation
    # ------------------------------

    def __init__(self, file_path: str, channel_index: int = 0):
        """
        Create a new WavIterator object

        :param file_path: Path to the WAV file
        :param channel_index: Index of the audio channel to process

        :raises ValueError: If the channel index is out of range
        """

        self._file_path = file_path

        self._window_index = 0
        self._channel_index = channel_index

        with wave.open(file_path, 'rb') as wav_file:
            self._frame_rate = wav_file.getframerate()
            self._num_frames = wav_file.getnframes()
            self._sample_width = wav_file.getsampwidth()
            self._num_channels = wav_file.getnchannels()

            self._window_size_frames = self._frame_rate // 10

            if self._num_channels <= self._channel_index:
                raise ValueError(f"Channel index out of range: {self._channel_index}")

            frames = wav_file.readframes(self._num_frames)
            dtype = self.get_data_type()

            self._audio_data = np.frombuffer(frames, dtype=dtype).reshape(-1, self._num_channels)

    # ------------------------------
    # Class interaction
    # ------------------------------

    def invalidate(self) -> None:
        """
        Invalidate the iterator
        """

        self._window_index = 0
        self._invalidate()

    # ------------------------------
    # Simple getters
    # ------------------------------

    def random_access(self, index: int) -> np.ndarray:
        """
        Plain window access method

        :param index: Index of the window
        """

        return self._audio_data[
               index * self._window_size_frames:(index + 1) * self._window_size_frames,
               self._channel_index]


    def get_data_type(self) -> type:
        """
        Return the type of the samples

        :return: Type of the samples

        :raises ValueError: If the sample width is not supported
        """
        types: dict[int, type] = {
            1: np.int8,
            2: np.int16,
            4: np.int32
        }

        if self._sample_width in types:
            return types[self._sample_width]
        raise ValueError(f"Unsupported sample width: {self._sample_width}")

    def get_window_size(self) -> int:
        """
        Return the window size

        :return: Size of the window in frames
        """

        return self._window_size_frames

    def get_frame_rate(self) -> float:
        """
        Return the frame rate

        :return: Frame rate in Hz
        """

        return self._frame_rate

    def get_num_frames(self) -> int:
        """
        Return the number of frames

        :return: Number of frames
        """

        return self._num_frames

    def get_sample_width(self) -> int:
        """
        Return the sample width

        :return: Sample width in bytes
        """

        return self._sample_width

    def get_num_channels(self) -> int:
        """
        Return the number of channels

        :return: Number of channels
        """

        return self._num_channels

    def get_data(self) -> np.ndarray:
        """
        Return the audio data
        """

        return self._audio_data

        # ------------------------------
        # Simple setters
        # ------------------------------

    def set_window_size(self, window_size: int) -> None:
        """
        Set the window size and invalidate the iterator

        :param window_size: Size of the window in frames
        """

        self._window_size_frames = window_size
        self._invalidate()

    def set_channel_index(self, channel_index: int) -> None:
        """
        Set the channel index and invalidate the iterator

        :param channel_index: Index of the audio channel to process

        :raises ValueError: If the channel index is out of range
        """

        if self._num_channels <= channel_index:
            raise ValueError(f"Channel index out of range: {channel_index}")

        self._channel_index = channel_index
        self.invalidate()

    # ------------------------------
    # Abstract methods
    # ------------------------------

    @abstractmethod
    def _get_next(self) -> np.ndarray:
        """
        Return the next window of samples

        Used in the __next__ method
        To be implemented in the derived classes

        :return: Array of samples
        """

        return np.array([])

    @abstractmethod
    def _invalidate(self) -> None:
        """
        Invalidate the iterator state

        To be implemented in the derived classes
        """

        return

    # ------------------------------
    # Iterator protocol
    # ------------------------------

    def __iter__(self) -> 'WavIteratorBase':
        """
        Return the iterator object itself.

        :return: self
        """

        return self

    def __next__(self) -> np.array:
        """
        Return the next window of samples

        :return: Array of samples
        """

        return self._get_next().flatten()


class OverlappingWavIterator(WavIteratorBase):
    """
    Iterator over the samples of a WAV file

    The iterator provides a window of samples of a fixed size. The window is moved by half of its
    size at each iteration.
    """

    # ------------------------------
    # Class fields
    # ------------------------------

    _prev_last_sample: int

    # ------------------------------
    # Class creation
    # ------------------------------

    def __init__(self, file_path: str, channel_index: int = 0) -> None:
        """
        Create a new OverlappingWavIterator object
        """
        super().__init__(file_path, channel_index)
        self._prev_last_sample = 0

    # ------------------------------
    # Class interaction
    # ------------------------------

    def _get_next(self) -> np.ndarray:
        """
        Return the next window of samples or raise StopIteration if the end of the file is reached

        :return: Array of samples
        """

        start_point_offset = self._window_size_frames // 2

        start_point = start_point_offset * self._window_index
        end_point = start_point + self._window_size_frames

        if end_point > self._num_frames:
            end_point = self._num_frames
            start_point = max(0, end_point - self._window_size_frames)

        if end_point <= self._prev_last_sample:
            raise StopIteration
        self._prev_last_sample = end_point

        self._window_index += 1

        return self._audio_data[start_point:end_point, self._channel_index]

    def _invalidate(self) -> None:
        """
        Invalidate the iterator state
        """

        self._prev_last_sample = 0


class PlainWavIterator(WavIteratorBase):
    """
    Iterator over the samples of a WAV file

    Simplest possible iterator over the samples of a WAV file.
    It provides a window of samples of a fixed size.
    """

    # ------------------------------
    # Class interaction
    # ------------------------------

    def _get_next(self) -> np.ndarray:
        """
        Get the next window of samples or raise StopIteration if the end of the file is reached

        :return: Array of samples
        """

        start_point = self._window_size_frames * self._window_index

        if start_point >= len(self._audio_data[:, self._channel_index]):
            raise StopIteration

        end_point = min(start_point + self._window_size_frames,
                        len(self._audio_data[:, self._channel_index]))
        self._window_index += 1

        return self._audio_data[start_point:end_point, self._channel_index]

    def _invalidate(self) -> None:
        """
        Invalidate the iterator state

        No state to invalidate in this class
        """

        return


class FlattenWavIterator:
    """
    Iterator wrapper over multiple WAV file iterators

    This iterator wraps multiple WAV file iterators and returns the mean of the samples
    """

    # ------------------------------
    # Class fields
    # ------------------------------

    _file_path: str
    _iters: list[WavIteratorBase]

    # ------------------------------
    # Class creation
    # ------------------------------

    def __init__(self, file_path: str, window_length_seconds: float,
                 iterator_type: WavIteratorType) -> None:
        iters = [load_wav(file_path, 0, iterator_type)]
        num_channels = iters[0].get_num_channels()

        for i in range(1, num_channels):
            iters.append(load_wav(file_path, i, iterator_type))

        for it in iters:
            it.set_window_size(int(it.get_frame_rate() * window_length_seconds))

        self._file_path = file_path
        self._iters = iters

        if len(self._iters) < 1:
            raise ValueError("No channels found in the file!")

    # ------------------------------
    # Class interaction
    # ------------------------------

    def get_first_iter(self) -> WavIteratorBase:
        """
        Get the first iterator

        :return: First iterator
        """

        return self._iters[0]

    # ------------------------------
    # Iterator Protocol
    # ------------------------------

    def __iter__(self) -> any:
        """
        Return the iterator object itself.

        :return: self
        """

        return self.iterate()

    def iterate(self) -> Generator[np.ndarray, None, None]:
        """
        Return the next window of samples

        :return: Array of samples
        """

        for chunks in zip(*self._iters):
            stacked_array = np.stack(chunks, axis=0)
            meaned_array = np.mean(stacked_array, axis=0)

            dtype = stacked_array.dtype
            flat_chunk = meaned_array.flatten().astype(dtype)
            yield flat_chunk


def load_wav(file_path: str, channel_index: int = 0,
             iterator_type: WavIteratorType = WavIteratorType.PLAIN) -> WavIteratorBase:
    """
    Load a WAV file and return an iterator over the samples
    :param file_path: Path to the WAV file
    :param channel_index: Index of the audio channel to process
    :param iterator_type: Type of the iterator to use

    :return: WavIterator object
    """

    if iterator_type == WavIteratorType.PLAIN:
        return PlainWavIterator(file_path, channel_index)
    if iterator_type == WavIteratorType.OVERLAPPING:
        return OverlappingWavIterator(file_path, channel_index)
    raise ValueError(f"Unsupported iterator type: {iterator_type}")


def load_wav_with_window(file_path: str,
                         window_length_seconds: float = 0.1,
                         channel_index: int = 0,
                         iterator_type: WavIteratorType = WavIteratorType.PLAIN) -> WavIteratorBase:
    """
    Load a WAV file and return an iterator over the samples with a window size being a fraction
    of the frame rate
    :param file_path: Path to the WAV file
    :param window_length_seconds: Length of each window in seconds
    :param channel_index: Index of the audio channel to process
    :param iterator_type: Type of the iterator to use

    :return: WavIterator object
    """

    if iterator_type == WavIteratorType.PLAIN:
        iterator = PlainWavIterator(file_path, channel_index)
    elif iterator_type == WavIteratorType.OVERLAPPING:
        iterator = OverlappingWavIterator(file_path, channel_index)
    else:
        raise ValueError(f"Unsupported iterator type: {iterator_type}")

    iterator.set_window_size(int(iterator.get_frame_rate() * window_length_seconds))
    return iterator


def cut_file_to_plain_chunk_files(file_path: str, destination_dir: str,
                                  window_length_seconds: float,
                                  iterator_type: WavIteratorType) -> int:
    """
    Cut the WAV file into chunks and save them as separate files with index suffix:
    {FILE}_{INDEX}.wav
    """

    os.makedirs(destination_dir, exist_ok=True)
    it = FlattenWavIterator(file_path, window_length_seconds, iterator_type)
    first_iter = it.get_first_iter()
    counter = 0

    for index, chunk in enumerate(it):
        output_file = os.path.join(destination_dir,
            f"{os.path.splitext(os.path.basename(file_path))[0]}_{index:0>3}.wav")

        try:
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(first_iter.get_sample_width())
                wav_file.setframerate(first_iter.get_frame_rate())
                wav_file.setcomptype('NONE', 'not compressed')
                wav_file.setnframes(len(chunk))
                wav_file.writeframes(np.array(chunk).tobytes())
            counter += 1
        # pylint: disable=broad-except
        except Exception as e:
            print(f"Error while processing {output_file}: {e}")
            os.remove(output_file)
    return counter
