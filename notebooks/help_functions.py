from IPython.display import Audio, display
from src.pipeline.audio_data import AudioData

"""
    Function to display an audio in jupyter notebook.
"""
def display_audio(audio_data: AudioData) -> None:
    """
    Displays an audio signal as an audio player widget.

    Args:
        audio_data (AudioData): The audio data to display.
    """
    display(Audio(audio_data.audio_signal, rate=audio_data.sample_rate))