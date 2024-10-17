from enum import Enum
import numpy as np

class SpeechDetectionType(Enum):
    """
    Enum for different types of speech detection.
    Future types of speech detection can be added here and
    handled in the is_speech function
    """
    SILENCE = 1

def is_speech(audio: np.ndarray, sr: int, speech_detection_type: SpeechDetectionType) -> bool:
    """
    Detect if the audio contains speech. 
    @param audio: audio signal
    @param sr: sample rate
    @param speech_detection_type: Type of speech detection to perform
    """
    if speech_detection_type == SpeechDetectionType.SILENCE:
        return silence_detection(audio, sr, silence_tolerance=0.5)

    raise ValueError(f"Unsupported speech detection type: {speech_detection_type}")

def silence_detection(audio: np.ndarray, sr: int, silence_tolerance: float) -> bool:
    """
    Detect if the audio contains a significant amount of silence.
    
    @param audio: audio signal
    @param sr: sample rate
    @param silence_tolerance: percentage of silence in the audio (0 to 1 range)
    @return: True if the audio contains more silence than the silence_tolerance, otherwise False.
    """
    silence_threshold = 0.10  # amplitude close to 0 is considered silence.
    
    silent_samples = np.sum(np.abs(audio) < silence_threshold)
    total_samples = len(audio)
    silence_percentage = silent_samples / total_samples
    
    return silence_percentage >= silence_tolerance