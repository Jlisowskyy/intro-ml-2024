"""
Author: Jakub Pietrzak, 2024

This module provides functionality to load a model from a specified file path
and print its details.
"""

import argparse

from src.model_definitions import BasicCNN


def main():
    """
    Main function that loads a model from the specified file path and prints the model.
    """
    parser = argparse.ArgumentParser(description="Process a file path to load a model.")
    parser.add_argument("model_file_path", type=str, help="Path to the model file")
    args = parser.parse_args()

    print(f"File path received: {args.model_file_path}")
    model = BasicCNN.load_model(args.model_file_path)
    print(model)
