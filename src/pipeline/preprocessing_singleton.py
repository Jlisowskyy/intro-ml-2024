from threading import Lock

from sklearn.pipeline import Pipeline
from src.constants import NormalizationType
from src.pipeline.audio_accelerator import AudioAccelerator
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.pipeline.audio_pitcher import AudioPitcher
from src.pipeline.echo_injector import EchoInjector
from src.pipeline.noise_injector import NoiseInjector
from src.pipeline.spectrogram_generator import SpectrogramGenerator


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class PreprocessingSingleton(metaclass = SingletonMeta):

    def __init__(self):
        self.preprocesing_pipeline = Pipeline(steps=[
            ('AudioCleaner', AudioCleaner()),
            ('AudioNormalizer', AudioNormalizer()),
            #('SpectrogramGenerator', SpectrogramGenerator())
        ])
        self.noise_injector = NoiseInjector()
        self.audio_accelerator = AudioAccelerator()
        self.audio_pitcher = AudioPitcher()
        self.echo_injector = EchoInjector()

    def inject_noise(self, audio_data: list[AudioData]):
        noised_audio = self.noise_injector.transform(audio_data)
        transformed_data = self.preprocesing_pipeline.transform(noised_audio)
        return transformed_data

    def accelerate_audio(self, audio_data: list[AudioData]):
        accelerated_audio = self.audio_accelerator.transform(audio_data)
        transformed_data = self.preprocesing_pipeline.transform(accelerated_audio)
        return transformed_data

    def pitch_audio(self, audio_data: list[AudioData]):
        pitched_audio = self.audio_pitcher.transform(audio_data)
        transformed_data = self.preprocesing_pipeline.transform(pitched_audio)
        return transformed_data

    def inject_echo(self, audio_data: list[AudioData]):
        echo_audio = self.echo_injector.transform(audio_data)
        transformed_data = self.preprocesing_pipeline.transform(echo_audio)
        return transformed_data
