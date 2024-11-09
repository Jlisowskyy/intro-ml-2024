"""
Author: Jakub Lisowski, 2024

Simple script runner
"""

import sys
import inspect
from typing import Callable

from src.helper_scripts import data_analysis
from src.helper_scripts import from_wav_to_histogram
from src.helper_scripts import generate_csv
from src.helper_scripts import generate_rgb_histogram
from src.helper_scripts import prepare_dataset
from src.helper_scripts import regenerate_csv
from src.helper_scripts import spectrogram_script
from src.helper_scripts import validate_dataset

SCRIPTS: dict[str, Callable[..., None]] = {
    "data_analysis": data_analysis.main,
    "from_wav_to_histogram": from_wav_to_histogram.main,
    "generate_csv": generate_csv.main,
    "generate_rgb_histogram": generate_rgb_histogram.main,
    "prepare_dataset": prepare_dataset.main,
    "regenerate_csv": regenerate_csv.main,
    "spectrogram_script": spectrogram_script.main,
    "validate_dataset": validate_dataset.main,
}


def validate_scripts() -> None:
    """
    Validate that each function in SCRIPTS has a correct signature
    (no arguments or a single argument of type list[str]).
    """
    for name, func in SCRIPTS.items():
        signature = inspect.signature(func)
        params = list(signature.parameters.values())

        if not (len(params) == 0 or
                (len(params) == 1 and params[0].annotation == list[str])):
            raise TypeError(f"Function '{name}' must take no arguments or a single argument of type list[str].")


def display_help() -> None:
    """
    Display help message
    """
    print("Usage: python script_runner.py <script_name>")
    print("Available scripts:")

    for script_name, func in SCRIPTS.items():
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        if len(params) == 1 and params[0].annotation == list[str]:
            print(f"\t{script_name} <args...>")
        else:
            print(f"\t{script_name}")


def run_script(script_name: str, args: list[str]) -> None:
    """
    Run the script with the given name or display help message if the script is not found
    """
    if script_name in SCRIPTS:
        func = SCRIPTS[script_name]
        signature = inspect.signature(func)
        params = list(signature.parameters.values())

        if len(params) == 0:
            func()
        elif len(params) == 1 and params[0].annotation == list[str]:
            func(args)
        else:
            raise TypeError(f"Function '{script_name}' has an invalid signature.")
    else:
        display_help()


def main() -> None:
    """
    Program entry point
    """
    validate_scripts()

    if len(sys.argv) < 2:
        display_help()
        sys.exit(1)

    arg = sys.argv[1]
    run_script(arg, sys.argv[2:])


if __name__ == '__main__':
    main()
