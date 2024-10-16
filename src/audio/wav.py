"""
Author: Jakub Lisowski, 2024

This module provides a simple iterator over the samples of a WAV file.

"""

import wave

import numpy as np


class WavIterator:
    """
    Iterator over the samples of a WAV file

    The iterator provides a window of samples of a fixed size. The window is moved by half of its
    size at each iteration.
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

    _prev_last_sample: int

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
        self._prev_last_sample = 0
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

    # ------------------------------
    # Simple getters
    # ------------------------------

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

        # ------------------------------
        # Simple setters
        # ------------------------------

    def set_window_size(self, window_size: int) -> None:
        """
        Set the window size and invalidate the iterator

        :param window_size: Size of the window in frames
        """

        self._window_size_frames = window_size
        self._window_index = 0
        self._prev_last_sample = 0

    def set_channel_index(self, channel_index: int) -> None:
        """
        Set the channel index and invalidate the iterator

        :param channel_index: Index of the audio channel to process

        :raises ValueError: If the channel index is out of range
        """

        if self._num_channels <= channel_index:
            raise ValueError(f"Channel index out of range: {channel_index}")

        self._channel_index = channel_index
        self._window_index = 0
        self._prev_last_sample = 0

    # ------------------------------
    # Iterator protocol
    # ------------------------------

    def __iter__(self) -> 'WavIterator':
        """
        Return the iterator object itself.

        :return: self
        """

        return self

    def __next__(self) -> np.ndarray:
        """
        Return the next window of samples

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


def load_wav(file_path: str, channel_index: int = 0) -> WavIterator:
    """
    Load a WAV file and return an iterator over the samples
    :param file_path: Path to the WAV file
    :param channel_index: Index of the audio channel to process

    :return: WavIterator object
    """

    return WavIterator(file_path, channel_index)


def load_wav_with_window(file_path: str, window_length_seconds: float = 0.1, channel_index: int = 0) \
        -> WavIterator:
    """
    Load a WAV file and return an iterator over the samples with a window size being a fraction
    of the frame rate
    :param file_path: Path to the WAV file
    :param window_length_seconds: Length of each window in seconds
    :param channel_index: Index of the audio channel to process

    :return: WavIterator object
    """

    iterator = WavIterator(file_path, channel_index)
    iterator.set_window_size(int(iterator.get_frame_rate() * window_length_seconds))

    return iterator
