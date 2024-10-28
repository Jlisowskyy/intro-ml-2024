import torch

from src.audio.audio_data import AudioData
from src.audio.denoise import denoise
from src.audio.normalize import normalize
from src.audio.spectrogram import gen_mel_spectrogram, save_spectrogram
from src.cnn.cnn import BasicCNN
from src.constants import NORMALIZATION_TYPE, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT
import soundfile as sf

chunk, sr=sf.read("C:/Users/pietr/Desktop/Nowy folder/f1_script4_iphone_balcony1.wav")

spectrogram = gen_mel_spectrogram(chunk, int(sr),
                                      width=SPECTROGRAM_WIDTH,
                                      height=SPECTROGRAM_HEIGHT)
save_spectrogram(spectrogram, "C:/Users/pietr/Desktop/Nowy folder/zaszumiony.png")


chunk = denoise(chunk, sr)
chunk = normalize(chunk, sr, NORMALIZATION_TYPE)
spectrogram = gen_mel_spectrogram(chunk, int(sr),
                                      width=SPECTROGRAM_WIDTH,
                                      height=SPECTROGRAM_HEIGHT)

save_spectrogram(spectrogram, "C:/Users/pietr/Desktop/Nowy folder/odszumiony.png")