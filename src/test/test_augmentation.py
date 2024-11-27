"""
Author: Michał Kwiatkowski

This module demonstrates a pipeline for audio processing and spectrogram generation.
It utilizes classes and methods for noise injection, acceleration, pitch modification,
echo injection, and cleaning of audio files. The processed audio data is then saved as
spectrogram images.
"""

from src.constants import DEFAULT_FILE_NAMES, TEST_FOLDER_IN, TEST_FOLDER_OUT, WavIteratorType
from src.pipeline.audio_data import AudioData
from src.pipeline.preprocessing_singleton import PreprocessingSingleton
from src.pipeline.spectrogram_generator import SpectrogramGenerator
from src.pipeline.wav import load_wav
from src.test.test_file import TestFile


# Define a test file for processing
TEST_FILE = TestFile(
    str(TEST_FOLDER_IN / DEFAULT_FILE_NAMES[2]),
    DEFAULT_FILE_NAMES[2],
    str(TEST_FOLDER_OUT / DEFAULT_FILE_NAMES[2])
)


def main() -> None:
    """
    Main function to demonstrate audio preprocessing and spectrogram generation.

    Steps:
    - Load and prepare audio data for processing.
    - Apply various preprocessing transformations such as:
        - Noise injection
        - Audio acceleration
        - Pitch adjustment
        - Echo injection
        - Audio cleaning
    - Generate and save spectrograms of the processed audio data.
    """

    # Initialize the preprocessing pipeline
    preprocessing_pipeline = PreprocessingSingleton()

    # Load the original and transformed audio files
    it_transformed = load_wav(TEST_FILE.file_path, 0, WavIteratorType.PLAIN)
    it_transformed.set_window_size(it_transformed.get_num_frames())

    it = load_wav(TEST_FILE.file_path, 0, WavIteratorType.PLAIN)
    it.set_window_size(it.get_num_frames())

    # Retrieve the audio data
    original_audio = next(iter(it))

    # Wrap the original audio data into an AudioData object
    original_audio_data = AudioData(original_audio, int(it.get_frame_rate()))

    # Apply preprocessing transformations
    injected_noise = preprocessing_pipeline.inject_noise([original_audio_data])
    accelerated_audio = preprocessing_pipeline.accelerate_audio([original_audio_data])
    pitched_audio = preprocessing_pipeline.pitch_audio([original_audio_data])
    injected_echo = preprocessing_pipeline.inject_echo([original_audio_data])
    cleaned_audio = preprocessing_pipeline.clean_audio([original_audio_data])

    # Generate and save spectrograms for the processed audio
    SpectrogramGenerator.save_spectrogram(
        cleaned_audio[0],
        TEST_FILE.get_transformed_file_path_out("cleaned").replace(".wav", ".png")
    )
    SpectrogramGenerator.save_spectrogram(
        injected_noise[0],
        TEST_FILE.get_transformed_file_path_out("injected_noise").replace(".wav", ".png")
    )
    SpectrogramGenerator.save_spectrogram(
        accelerated_audio[0],
        TEST_FILE.get_transformed_file_path_out("accelerated_audio").replace(".wav", ".png")
    )
    SpectrogramGenerator.save_spectrogram(
        pitched_audio[0],
        TEST_FILE.get_transformed_file_path_out("pitched_audio").replace(".wav", ".png")
    )
    SpectrogramGenerator.save_spectrogram(
        injected_echo[0],
        TEST_FILE.get_transformed_file_path_out("injected_echo").replace(".wav", ".png")
    )
