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

    PAGE_PATH: Path = Path.resolve(Path(f'{__file__}/../index.html'))

    page: str
    classifier: BasicCNN

    def __init__(self) -> None:
        """
        Initialize the application state
        """

        with open(AppState.PAGE_PATH, 'r', encoding='utf-8') as f:
            self.page = f.read()

        self.classifier = BasicCNN.load_model(MODEL_BASE_PATH)
