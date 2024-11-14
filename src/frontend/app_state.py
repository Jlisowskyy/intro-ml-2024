"""
Author: Jakub Lisowski, 2024

State of the application
"""

from pathlib import Path

from src.cnn.cnn import BasicCNN
from src.constants import MODEL_BASE_PATH


class AppState:
    """
    Class representing the state of the application
    """

    index_path: Path
    page: str
    classifier: BasicCNN

    def __init__(self):
        """
        Initialize the application state
        """

        self.index_path = Path.resolve(Path(f'{__file__}/../index.html'))

        with open(self.index_path, 'r', encoding='utf-8') as f:
            self.page = f.read()

        self.classifier = BasicCNN.load_model(MODEL_BASE_PATH)
