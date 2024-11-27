import wave
from io import BytesIO

# General AudioData class
class AudioData:
    def __init__(self, audio_data: bytes):
        self.audio_data = audio_data

    def is_valid(self) -> bool:
        """To be overridden by child classes for specific audio type validation."""
        raise NotImplementedError("Subclasses should implement this!")

    def get_properties(self) -> dict:
        """To be overridden by child classes to extract audio properties."""
        raise NotImplementedError("Subclasses should implement this!")

# Specialized WaveData class for .wav files
class WaveData(AudioData):
    def __init__(self, audio_data: bytes):
        super().__init__(audio_data)
        self.wav_file = None

    def is_valid(self) -> bool:
        """Check if the audio data is a valid WAV file."""
        try:
            self.wav_file = wave.open(BytesIO(self.audio_data), "rb")
            return True
        except wave.Error:
            return False

    def get_properties(self) -> dict:
        """Extract properties from the WAV file."""
        if not self.wav_file:
            raise ValueError("WAV file not opened. Ensure the file is valid first.")

        return {
            "channels": self.wav_file.getnchannels(),
            "sample_width": self.wav_file.getsampwidth(),
            "frame_rate": self.wav_file.getframerate(),
            "n_frames": self.wav_file.getnframes(),
        }
