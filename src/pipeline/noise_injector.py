"""
Author: Michał Kwiatkowski
"""

from src.pipeline.audio_data import AudioData


class NoiseInjector:

    def __init__(self) -> None:
        return

    # pylint: disable=unused-argument
    def fit(self, x_data: list[AudioData], y_data: list[int] = None) -> 'NoiseInjector':
        return

    def transform(self, x_data: list[AudioData]) -> list[AudioData]:
        return
