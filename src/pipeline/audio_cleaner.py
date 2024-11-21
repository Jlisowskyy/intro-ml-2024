"""
Author: Michał Kwiatkowski, Łukasz Kryczka

This module contains the AudioCleaner class, which is used to clean audio data
as part of a machine learning pipeline. It implements the fit and transform
methods to be compatible with scikit-learn pipelines. It provides
functionality for denoising WAV data using simple filters. Currently, supports
basic denoising for human speech frequencies.
"""

from denoiser import pretrained
import torch
import torchaudio
import numpy as np

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
