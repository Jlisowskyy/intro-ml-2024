"""
Author: Michał Kwiatkowski, Łukasz Kryczka, Jakub Lisowski

This module contains the AudioCleaner class, which is used to clean audio data
as part of a machine learning pipeline. It implements the fit and transform
methods to be compatible with scikit-learn pipelines. It provides
functionality for denoising WAV data using simple filters. Currently, supports
basic denoising for human speech frequencies.
"""

import numpy as np
import torch
import torchaudio
from denoiser import pretrained

from src.constants import DETECT_SILENCE_THRESHOLD_DB, DETECT_SILENCE_WINDOW_MAX_MS, \
    SILENCE_CUT_WINDOW_MS
from src.pipeline.audio_data import AudioData


class AudioCleaner:
    """
    A class used to clean audio data as part of a machine learning pipeline.
    """

    def __init__(self) -> None:
        """
        Initializes the AudioCleaner.

        Parameters:
        denoise_type (DenoiseType): Type of denoising to perform (from DenoiseType enum).

        Returns:
        None
        """
        self.denoiser_model = pretrained.dns64()
        self.denoiser_model.eval()
        self.denoiser_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def denoise(self, audio_data: AudioData) -> AudioData:
        """
        Denoise the given audio data using the pretrained DNS64 model.
        More info on the model: https://pypi.org/project/denoiser/

        :param audio_data: Audio data to be denoised.
        :return: Denoised chunk of audio data.
        """
        with torch.no_grad():
            # Denoiser requires mono .wav files with 16 kHz sample rate

            # Ensure the signal is scaled to [-1, 1]
            audio_signal = audio_data.audio_signal
            if audio_signal.dtype != np.float32:
                audio_signal = AudioData.to_float(audio_signal)

            # Resample to 16 kHz if needed
            if audio_data.sample_rate != 16000:
                resampled_signal = torchaudio.transforms.Resample(
                    orig_freq=audio_data.sample_rate, new_freq=16000
                )(torch.tensor(audio_signal, dtype=torch.float32))
            else:
                resampled_signal = torch.tensor(audio_signal, dtype=torch.float32)

            # Add batch and channel dimensions
            if resampled_signal.ndim == 1:  # Mono audio
                resampled_signal = resampled_signal.unsqueeze(0)  # Add batch dimension
            elif resampled_signal.ndim == 2 and resampled_signal.shape[0] > 2:  # Stereo audio
                resampled_signal = resampled_signal.T.unsqueeze(0)  # Correct channel order

            device = next(self.denoiser_model.parameters()).device
            resampled_signal = resampled_signal.to(device)

            denoised_signal = self.denoiser_model(resampled_signal)[0]

            # Remove batch and channel dimensions
            denoised_signal = denoised_signal.squeeze(0).cpu().numpy()

            # Resample back to original sample rate if needed
            if audio_data.sample_rate != 16000:
                denoised_signal = torchaudio.transforms.Resample(
                    orig_freq=16000, new_freq=audio_data.sample_rate
                )(torch.tensor(denoised_signal, dtype=torch.float32)).numpy()

            audio_data.audio_signal = denoised_signal

        return audio_data

    @staticmethod
    def is_speech(audio_data: AudioData, silence_threshold=DETECT_SILENCE_THRESHOLD_DB) -> bool:
        """
        Determines if an audio segment contains speech based on RMS energy threshold.

        Parameters:
            audio_data (AudioData): Audio segment to analyze.
            silence_threshold (float): RMS energy threshold in dB to classify as speech.

        Returns:
            bool: True if segment contains speech, False if classified as silence.
        """
        duration = len(audio_data.audio_signal) / audio_data.sample_rate
        assert (duration * 1000) < DETECT_SILENCE_WINDOW_MAX_MS

        rms = np.sqrt(np.mean(np.square(audio_data.audio_signal)))
        rms_db = 20 * np.log10(rms + 1e-9)

        return rms_db > silence_threshold

    @staticmethod
    def remove_silence_raw(audio_data: np.ndarray, frame_rate: int,
                           silence_threshold: int = DETECT_SILENCE_THRESHOLD_DB) -> np.ndarray:
        """
        Removes silent segments from raw audio data by analyzing fixed-size windows.
        Expects and returns audio data in shape (N, 1).

        Parameters:
            audio_data (np.ndarray): Raw audio signal data in shape (N, 1).
            frame_rate (int): Audio sample rate in Hz.
            silence_threshold (int): RMS energy threshold in dB to classify as speech.

        Returns:
            np.ndarray: Audio data with silent segments removed, maintaining shape (N, 1).

        Raises:
            ValueError: If the input audio is not in shape (N, 1).
        """
        
        if len(audio_data.shape) != 2 or audio_data.shape[1] != 1:
            raise ValueError(f"Input audio must be in shape (N, 1). Got shape: {audio_data.shape}")

        audio_1d = audio_data.flatten()
        audio = AudioData(audio_1d, frame_rate)
        processed_audio = AudioCleaner.remove_silence(audio, silence_threshold)
        return processed_audio.audio_signal.reshape(-1, 1)

    @staticmethod
    def remove_silence(audio_data: AudioData,
                       silence_threshold: int = DETECT_SILENCE_THRESHOLD_DB) -> AudioData:
        """
        Removes silent segments from an AudioData object.

        Parameters:
            audio_data (AudioData): Audio data to process.
            silence_threshold (float): RMS energy threshold in dB to classify as speech.

        Returns:
            AudioData: New AudioData object with silent segments removed.
        """
        window = (SILENCE_CUT_WINDOW_MS * audio_data.sample_rate) // 1000

        output = np.array([], dtype=audio_data.audio_signal.dtype)

        for i in range(0, len(audio_data.audio_signal), window):
            chunk_data = audio_data.audio_signal[i:i + window]
            chunk = AudioData(chunk_data, audio_data.sample_rate)

            if AudioCleaner.is_speech(chunk, silence_threshold):
                output = np.concatenate((output, chunk_data))

        return AudioData(output, audio_data.sample_rate)


    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'AudioCleaner':
        """
        Fits the transformer to the data.

        Parameters:
        x_data (array-like): Input data - list of AudioData objects.
        y_data (array-like, optional): Target values (ignored in this transformer).

        Returns:
        self: Returns the instance itself.
        """
        # No fitting process for this cleaner, but required for sklearn pipeline compatibility
        return self

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        """
        Transforms the input data by cleaning the audio.

        Parameters:
        x_data (array-like): Input data - list of AudioData objects.

        Returns:
        x_data (array-like): Output data - list of cleaned AudioData objects.
        """

        transformed_data = []
        for audio_data in x_data:
            transformed_data.append(self.denoise(audio_data))

        return transformed_data
