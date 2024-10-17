from sklearn.pipeline import Pipeline
import LoadWav
import AudioCleaner
import AudioNormalizer
from SpectogramGenerator import SpectogramGenerator
import Classifier
from AudioData import AudioData
import os
from glob import glob
import soundfile as sf
import numpy as np

def main():
    # training_pipeline = Pipeline(steps=[
    #     ('AudioCleaner', AudioCleaner()),
    #     ('AudioNormalizer', AudioNormalizer()),
    #     ('SpectogramGenerator', SpectogramGenerator()),
    #     ('Classifier', Classifier())
    # ])

    speaker_to_class = {
        'f1': 1,
        'f7': 1,
        'f8': 1,
        'm3': 1,
        'm6': 1,
        'm8': 1
    }

    audio_directory_path = "/home/michal/studia/sem5/ml/daps/clean"

    wav_files = glob(os.path.join(audio_directory_path, "*.wav"))

    X_train = []
    Y_train = []
    
    if not wav_files:
        print("No .wav files found in the directory.")
    else:
        print(f"Found {len(wav_files)} .wav files.")
        for wav_file_path in wav_files:
            audio_data_wav, sample_rate = sf.read(wav_file_path)
            audio_data = AudioData(np.array(audio_data_wav), sample_rate)

            speaker = wav_file_path.split('_')[0]
            speaker_class = speaker_to_class.get(speaker.lower(), 0)

            X_train.append(audio_data)
            Y_train.append(speaker_class)

    spectogram_generator = SpectogramGenerator()
    spectogram_data = spectogram_generator.transform(X_train)

    print('test')

if __name__ == "__main__":
    main()