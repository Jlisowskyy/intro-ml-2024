"""
Author: Jakub Lisowski, 2024

Simple test starting script
"""

import sys
from typing import Callable

import pytest

from src.test import test_cnn
from src.test import test_cut_wav
from src.test import test_denoise
from src.test import test_normalize
from src.test import test_validator
from src.test import test_wav

def run_pytest() -> None:
    """
    Run pytest
    """

    pytest.main(["-v", "src/test"])


TEST_CASES: dict[str, Callable[[], None]] = {
    "wav": test_wav.manual_test,
    "pytest": run_pytest,
    "validator": test_validator.manual_test,
    "denoise": test_denoise.manual_test,
    "normalize": test_normalize.manual_test,
    "cut_wav": test_cut_wav.manual_test,
    "cnn": test_cnn.manual_test
}


def display_help() -> None:
    """
    Display help message
    """

    print("Usage: python test_runner.py <test_name>")
    print("Available tests:")
    for test_name in TEST_CASES:
        print(f"  {test_name}")


def run_test(test_name: str) -> None:
    """
    Run the test with the given name or display help message if the test is not found
    """

    if test_name in TEST_CASES:
        TEST_CASES[test_name]()
    else:
        display_help()


def main() -> None:
    """
    Program entry point
    """

    if len(sys.argv) != 2:
        display_help()
        sys.exit(1)

    arg = sys.argv[1]

    run_test(arg)


if __name__ == '__main__':
    main()
