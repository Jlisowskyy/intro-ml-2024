"""
Author: Jakub Lisowski, 2024

File containing all the validation objects manual tests
"""

import numpy as np

from src.validation.false_acceptance_ratio import FalseAcceptanceRatio
from src.validation.false_rejection_ratio import FalseRejectionRatio
from src.validation.validator import Validator

TEST_CHUNKS = 1000


def test_classifier(_: np.ndarray) -> int:
    """
    Simulates classifier by returning random result
    """

    return np.random.randint(2)


def false_acceptance_ratio_test() -> None:
    """
    Simply uses simulated classifier to test the False Acceptance Ratio by validating random data
    """

    print("False Acceptance Ratio test: ")
    validator = Validator(test_classifier, [FalseAcceptanceRatio])

    for _ in range(TEST_CHUNKS):
        validator.validate((np.ndarray([]), 0))

    validator.display_textual_result()
    validator.display_graphical_result()


def false_rejection_ratio_test() -> None:
    """
    Simply uses simulated classifier to test the False Rejection Ratio by validating random data
    """

    print("False Rejection Ratio test: ")
    validator = Validator(test_classifier, [FalseRejectionRatio])

    for _ in range(TEST_CHUNKS):
        validator.validate((np.ndarray([]), 1))

    validator.display_textual_result()
    validator.display_graphical_result()


def validator_test() -> None:
    """
    Simply uses simulated classifier to test the Validator by validating random data

    Uses multiple validation objects
    """

    print("Validator test: ")
    validator = Validator(test_classifier, [FalseRejectionRatio, FalseAcceptanceRatio])

    for _ in range(TEST_CHUNKS):
        validator.validate((np.ndarray([]), np.random.randint(2)))

    validator.display_textual_result()
    validator.display_graphical_result()


def manual_test() -> None:
    """
    Run all the validation objects tests
    """

    false_acceptance_ratio_test()
    print("-" * 50)
    false_rejection_ratio_test()
    print("-" * 50)
    validator_test()
