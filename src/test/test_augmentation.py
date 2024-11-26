"""
Author: Michał Kwiatkowski
"""


from scipy.io.wavfile import write
from src.constants import DEFAULT_FILE_NAMES, TEST_FOLDER_IN, TEST_FOLDER_OUT, WavIteratorType
from src.pipeline.audio_data import AudioData
from src.pipeline.preprocessing_singleton import PreprocessingSingleton
from src.pipeline.wav import load_wav
from src.test.test_file import TestFile


TEST_FILE = TestFile(
        str(TEST_FOLDER_IN / DEFAULT_FILE_NAMES[2]),
        DEFAULT_FILE_NAMES[2],
        str(TEST_FOLDER_OUT / DEFAULT_FILE_NAMES[2])
    )


def main() -> None:
    preprocessing_pipeline = PreprocessingSingleton()

    it_transformed = load_wav(TEST_FILE.file_path, 0, WavIteratorType.PLAIN)
    it_transformed.set_window_size(it_transformed.get_num_frames())

    it = load_wav(TEST_FILE.file_path, 0, WavIteratorType.PLAIN)
    it.set_window_size(it.get_num_frames())

    original_audio = next(iter(it))
    processed_audio = next(iter(it_transformed))

    original_audio_data = AudioData(original_audio, int(it.get_frame_rate()))
    processed_audio_data = AudioData(processed_audio, int(it.get_frame_rate()))

    injected_noise = preprocessing_pipeline.inject_noise([processed_audio_data])
    accelerated_audio = preprocessing_pipeline.accelerate_audio([processed_audio_data])
    pitched_audio = preprocessing_pipeline.pitch_audio([processed_audio_data])
    injected_echo = preprocessing_pipeline.inject_echo([processed_audio_data])

    write(TEST_FILE.get_transformed_file_path_out("input"),
                original_audio_data.sample_rate, original_audio_data.audio_signal)

    write(TEST_FILE.get_transformed_file_path_out("injected_noise"),
                injected_noise[0].sample_rate, injected_noise[0].audio_signal)

    write(TEST_FILE.get_transformed_file_path_out("accelerated_audio"),
                accelerated_audio[0].sample_rate, accelerated_audio[0].audio_signal)

    write(TEST_FILE.get_transformed_file_path_out("pitched_audio"),
                pitched_audio[0].sample_rate, pitched_audio[0].audio_signal)

    write(TEST_FILE.get_transformed_file_path_out("injected_echo"),
                injected_echo[0].sample_rate, injected_echo[0].audio_signal)
