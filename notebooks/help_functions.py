"""
    Functions that help in notebook.
"""

from IPython.display import Audio, display
from src.pipeline.audio_data import AudioData

def display_audio(audio_data: AudioData) -> None:
    """
    Displays an audio signal as an audio player widget.

    Args:
        audio_data (AudioData): The audio data to display.
    """
    display(Audio(audio_data.audio_signal, rate=audio_data.sample_rate))
    