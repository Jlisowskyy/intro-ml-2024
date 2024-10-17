import numpy as np

class AudioData:
    def __init__(self, audio_signal: np.array, sample_rate: int):
        self.audio_signal = audio_signal
        self.sample_rate = sample_rate
        