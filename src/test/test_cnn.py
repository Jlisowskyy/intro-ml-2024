"""
Author: Jakub Lisowski, 2024

Tests for CNN model
"""

from torchsummary import summary

from src.model_definitions import BasicCNN
from src.cnn.loadset import DAPSDataset
from src.constants import DATABASE_ANNOTATIONS_PATH, DATABASE_OUT_PATH, SPECTROGRAM_WIDTH, \
    SPECTROGRAM_HEIGHT

PNG_NUM_COLORS = 3


def manual_test_dataset() -> None:
    """
    Manual test for the DAPSDataset
    """
    d = DAPSDataset(DATABASE_ANNOTATIONS_PATH, DATABASE_OUT_PATH,
                    'cuda')
    print(d[0])


def manual_test_cnn() -> None:
    """
    Manual test for the CNN model
    """

    cnn = BasicCNN().to('cuda')
    summary(cnn, (PNG_NUM_COLORS, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT))
