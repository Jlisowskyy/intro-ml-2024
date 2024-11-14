"""
Author: Jakub Pietrzak, 2024

This module provides functionality to load a model from a specified file path
and print its details.
"""

import argparse
from src.cnn.cnn import BasicCNN

def main(model_file_path: str):
    """
    Main function that loads a model from the specified file path and prints the model.

    Args:
        model_file_path (str): The file path to the saved model.
    """
    print(f"File path received: {model_file_path}")

    model = BasicCNN.load_model(model_file_path)
    print(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file path to load a model.")
    parser.add_argument("model_file_path", type=str, help="Path to the model file")

    args = parser.parse_args()
    main(args.model_file_path)
