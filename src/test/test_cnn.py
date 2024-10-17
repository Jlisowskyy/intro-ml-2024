"""
Author: Jakub Lisowski, 2024

Tests for CNN model
"""

from torchsummary import summary

from src.cnn.cnn import CNN


def display_summary() -> None:
    cnn = CNN()
    summary(cnn.cuda(), (1, 64, 157))


def manual_test() -> None:
    display_summary()
