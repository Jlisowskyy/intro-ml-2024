"""
Author: Jakub Lisowski, 2024

ValidatorObj implementation of the False Rejection Ratio (FRR) validation object
"""

from typing import Callable

import numpy as np

from src.validation.validator import ValidatorObj


class FalseRejectionRatio(ValidatorObj):
    """
    False Rejection Ratio (FRR) validation object

    The False Rejection Ratio (FRR) is a measure of the rate at which the classifier incorrectly
    rejects an input that should have been accepted.
    """

    # ------------------------------
    # Class fields
    # ------------------------------

    _false_rejection_count: int
    _total_count: int

    # ------------------------------
    # Class creation
    # ------------------------------

    def __init__(self, classifier: Callable[[np.ndarray], int]) -> None:
        """
        Create a new FalseRejectionRatio object

        :param classifier: The classifier function to validate
        """

        super().__init__(classifier)

        self._false_rejection_count = 0
        self._total_count = 0

    # ------------------------------
    # Class interaction
    # ------------------------------

    def validate(self, input_data: tuple[np.ndarray, int]) -> None:
        """
        Save the validation result to the internal state

        :param input_data: The input data to validate the classifier [data, expected_output]
        """

        data, expected_output = input_data

        if expected_output == 0:
            return

        self._total_count += 1
        if self._classifier(data) != expected_output:
            self._false_rejection_count += 1

    def display_graphical_result(self) -> None:
        """
        Display the validation result in a graphical form
        """

        pass

    def get_textual_result(self) -> str:
        """
        Get the validation result in a textual form

        :return: The textual representation of the validation result
        """

        return f"False Rejection Ratio: {self._false_rejection_count / self._total_count}"
