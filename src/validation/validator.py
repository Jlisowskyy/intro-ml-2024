"""
Author: Jakub Lisowski, 2024

Mail validation function
"""

from abc import ABC, abstractmethod
from typing import Callable, Type

import numpy as np


class ValidatorObj(ABC):
    """
    ValidatorObj class

    Base class for all various validation objects that will be used to validate the classifier
    """

    # ------------------------------
    # Class fields
    # ------------------------------

    _classifier: Callable[[np.ndarray], int]

    # ------------------------------
    # Class creation
    # ------------------------------

    def __init__(self, classifier: Callable[[np.ndarray], int]) -> None:
        """
        Create a new ValidatorObj object

        :param classifier: The classifier function to validate
        """
        self._classifier = classifier

    # ------------------------------
    # Class abstract methods
    # ------------------------------

    @abstractmethod
    def validate(self, input_data: tuple[np.ndarray, int]) -> None:
        """
        Save the validation result to the internal state

        :param input_data: The input data to validate the classifier [data, expected_output]
        """
        pass

    @abstractmethod
    def display_graphical_result(self) -> None:
        """
        Display the validation result in a graphical form
        """
        pass

    @abstractmethod
    def get_textual_result(self) -> str:
        """
        Get the validation result in a textual form
        """
        pass


class Validator:
    """
    Validator class

    Class used to validate the classifier by measuring various metrics and displaying them at the end

    :param classifier: The classifier function to validate
    """

    # ------------------------------
    # Class fields
    # ------------------------------

    _classifier: Callable[[np.ndarray], int]
    _validator_objects: list[ValidatorObj]

    # ------------------------------
    # Class creation
    # ------------------------------

    def __init__(self, classifier: Callable[[np.ndarray], int],
                 validator_objects: list[Type[ValidatorObj]]) -> None:
        """
        Create a new Validator object

        :param classifier: The classifier function to validate
        :param validator_objects: List of validator objects to use
        """

        self._classifier = classifier
        self._validator_objects = [validator(classifier) for validator in validator_objects]

    # ------------------------------
    # Class interaction
    # ------------------------------

    def validate(self, input_data: tuple[np.ndarray, int]) -> None:
        """
        Validate the classifier using the input data

        :param input_data: The input data to validate the classifier [data, expected_output]
        """

        for validator in self._validator_objects:
            validator.validate(input_data)

    def display_graphical_result(self) -> None:
        """
        Display the validation result in a graphical form
        """

        for validator in self._validator_objects:
            validator.display_graphical_result()

    def get_textual_result(self) -> list[str]:
        """
        Get the validation result in a textual form

        :return: List of textual results
        """

        return [validator.get_textual_result() for validator in self._validator_objects]

    def display_textual_result(self) -> None:
        """
        Display the validation result in a textual form in pretty format
        """

        for validator in self._validator_objects:
            print(f"Validator: {validator.__class__.__name__} result:")
            print("\t\t" + validator.get_textual_result().replace("\n", "\n\t\t"))
