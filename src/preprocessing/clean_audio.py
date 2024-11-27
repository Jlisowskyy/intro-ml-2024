import noisereduce as nr

from src.models.audio_data import AudioData

def clean_audio(audio_data : AudioData) -> AudioData:
    """Apply noise reduction to the audio data."""
    properties = audio_data.get_properties()
    reduced_noise = nr.reduce_noise(y=audio_data.audio_data, sr=properties["frame_rate"])
    audio_data.audio_data = reduced_noise
    return audio_data
