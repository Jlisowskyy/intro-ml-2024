"""
A module for audio preprocessing using a singleton class.

Provides `PreprocessingSingleton` for transformations such as noise injection, 
acceleration, pitch modification, echo injection, and audio cleaning. Uses a 
thread-safe singleton pattern and scikit-learn pipelines for structured audio processing.
"""

import threading
from typing import List
from sklearn.pipeline import Pipeline

from src.pipeline.audio_accelerator import AudioAccelerator
from src.pipeline.audio_cleaner import AudioCleaner
from src.pipeline.audio_data import AudioData
from src.pipeline.audio_normalizer import AudioNormalizer
from src.pipeline.audio_pitcher import AudioPitcher
from src.pipeline.echo_injector import EchoInjector
from src.pipeline.noise_injector import NoiseInjector
from src.pipeline.spectrogram_generator import SpectrogramGenerator

class SingletonMeta(type):
    """
    A thread-safe singleton metaclass that ensures only one instance of a class is created.
    
    This metaclass uses double-checked locking to create a single instance of a class,
    with thread safety to prevent race conditions during instantiation.
    """

    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Override the __call__ method to implement singleton pattern.
        
        Args:
            *args: Positional arguments for class instantiation
            **kwargs: Keyword arguments for class instantiation
        
        Returns:
            The single instance of the class
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class PreprocessingSingleton(metaclass=SingletonMeta):
    """
    A singleton class for managing audio preprocessing pipelines and transformations.
    
    This class provides methods for various audio transformations such as:
    - Noise injection
    - Audio acceleration
    - Pitch modification
    - Echo injection
    - Audio cleaning
    
    The preprocessing pipeline uses scikit-learn's Pipeline for consistent data transformation.
    """

    def __init__(self):
        """
        Initialize the preprocessing pipeline and various audio transformers.
        
        Sets up a pipeline with audio cleaning and normalization steps, 
        and initializes separate transformers for additional audio modifications.
        """
        # Create the base preprocessing pipeline
        self.preprocessing_pipeline = Pipeline(steps=[
            ('AudioCleaner', AudioCleaner()),
            ('AudioNormalizer', AudioNormalizer()),
            ('SpectrogramGenerator', SpectrogramGenerator())
        ])

        # Initialize individual audio transformers
        self.noise_injector = NoiseInjector()
        self.audio_accelerator = AudioAccelerator()
        self.audio_pitcher = AudioPitcher()
        self.echo_injector = EchoInjector()

    def inject_noise(self, audio_data: List[AudioData]):
        """
        Inject noise into audio data and apply preprocessing pipeline.
        
        Args:
            audio_data: A list of AudioData objects to be noise-injected
        
        Returns:
            Transformed AudioData after noise injection and preprocessing
        """
        noised_audio = self.noise_injector.transform(audio_data)
        transformed_data = self.preprocessing_pipeline.transform(noised_audio)
        return transformed_data

    def accelerate_audio(self, audio_data: List[AudioData]):
        """
        Accelerate audio data and apply preprocessing pipeline.
        
        Args:
            audio_data: A list of AudioData objects to be accelerated
        
        Returns:
            Transformed AudioData after acceleration and preprocessing
        """
        accelerated_audio = self.audio_accelerator.transform(audio_data)
        transformed_data = self.preprocessing_pipeline.transform(accelerated_audio)
        return transformed_data

    def pitch_audio(self, audio_data: List[AudioData]):
        """
        Modify pitch of audio data and apply preprocessing pipeline.
        
        Args:
            audio_data: A list of AudioData objects to have pitch modified
        
        Returns:
            Transformed AudioData after pitch modification and preprocessing
        """
        pitched_audio = self.audio_pitcher.transform(audio_data)
        transformed_data = self.preprocessing_pipeline.transform(pitched_audio)
        return transformed_data

    def inject_echo(self, audio_data: List[AudioData]):
        """
        Inject echo into audio data and apply preprocessing pipeline.
        
        Args:
            audio_data: A list of AudioData objects to have echo added
        
        Returns:
            Transformed AudioData after echo injection and preprocessing
        """
        echo_audio = self.echo_injector.transform(audio_data)
        transformed_data = self.preprocessing_pipeline.transform(echo_audio)
        return transformed_data

    def clean_audio(self, audio_data: List[AudioData]):
        """
        Apply preprocessing pipeline to clean audio data.
        
        Args:
            audio_data: A list of AudioData objects to be cleaned
        
        Returns:
            Transformed AudioData after preprocessing
        """
        transformed_data = self.preprocessing_pipeline.transform(audio_data)
        return transformed_data
