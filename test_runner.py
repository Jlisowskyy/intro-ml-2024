"""
Author: Jakub Lisowski, 2024

Simple test starting script
"""

import sys
from typing import Callable

import pytest

from src.test import test_wav


def run_pytest() -> None:
    """
    Run pytest
    """

    pytest.main(["-v", "src/test"])


TEST_CASES: dict[str, Callable[[], None]] = {
    "wav": test_wav.example_test_run,
    "pytest": run_pytest,
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
