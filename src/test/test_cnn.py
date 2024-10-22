"""
Author: Jakub Lisowski, 2024

Tests for CNN model
"""

from torchsummary import summary

from src.cnn.cnn import BasicCNN


def display_summary() -> None:
    """
    Display summary of the CNN model
    """

    cnn = BasicCNN()
    summary(cnn.cuda(), (3, 300, 400))


def manual_test() -> None:
    """
    Manual test for the CNN model
    """

    display_summary()
