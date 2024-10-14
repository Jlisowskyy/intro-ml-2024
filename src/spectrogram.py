"""
File for generating spectrogram
"""
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image




"""
Example use:
import soundfile as sf
audio_file_path="C:/Users/pietr/Documents/studia/machine learning/projekt/cutted_audios
    /class0/f2_script1_ipad_office1_5000.wav"
audio_data, sample_rate = sf.read(audio_file_path)
spectrogram=gen_spectrogram(audio_data, sample_rate)
save_spectogram(spectrogram,"./spectrogram.png")
"""

def gen_spectrogram(audio_data:np.array, sample_rate:int,
                    show_axis:bool=False, width:int=400, height:int=300) -> np.array:
    """
    Function generates mel-frequency spectrogram based on audio data. Audio data is a numpy array. 
    It retrurns numpy array that represents - 
    image of spectrogram.
    It uses Mel scale, which is more aligned with human hearing perception. This makes it effective 
    for speech or voice-based tasks.
    Mel scale emphasizes the lower frequencies where most of the speech information resides while 
    compressing the higher frequencies.
    This makes Mel spectrogram efficient for voice recognition and voice detection and emotion 
    detection.
    It underline sounds that human ear hears.
    Mel-frequency scale, which is a linear frequency space below 1000 Hz and a logarithmic space 
    above 1000 Hz.
    """
    dpi = 100
    fmax = 8000
    s = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate,
                                       n_fft=4096, hop_length=512, n_mels=512, fmax=fmax)
    s_db = librosa.power_to_db(s, ref=np.max)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    if show_axis:
        img = librosa.display.specshow(s_db, sr=sample_rate, fmax=fmax,
                                       x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel-Frequency Spectrogram')
    else:
        img = librosa.display.specshow(s_db, sr=sample_rate, fmax=fmax, ax=ax)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    image = Image.open(buf).convert('RGB')
    image_array = np.array(image)

    buf.close()
    plt.close(fig)

    return image_array

def save_spectogram(spectrogram, file_path):
    """
    Saves spectrogram to file
    """
    plt.imshow(spectrogram)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    
