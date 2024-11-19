"""
Author: Jakub Lisowski 2024

Command-line interface for neural network training, validation, database preparation and testing.
Provides both command-line argument parsing and interactive mode for executing operations.

Usage:
    Command line:
        python main.py [-t | -v | -r | -p]
        python main.py script <script_name> [args...]
        python main.py test <test_name>

    Interactive mode (when no arguments provided):
        Enter commands when prompted: train, validate, run, prepare, script, test

Adding new functionality:
    - Add new scripts to SCRIPTS dict with function reference
    - Add new tests to TEST_CASES dict with test function reference
    - Script functions should take no args or single list[str] parameter all
    - Test function should take no args
"""

import argparse
import inspect
from pathlib import Path
from typing import Callable

import pytest
import uvicorn

from src.cnn import train
from src.helper_scripts import data_analysis
from src.helper_scripts import from_wav_to_histogram
from src.helper_scripts import generate_rgb_histogram
from src.helper_scripts import prepare_dataset
from src.helper_scripts import regenerate_csv
from src.helper_scripts import spectrogram_script
from src.helper_scripts import validate_dataset
from src.test import test_cnn
from src.test import test_cut_wav
from src.test import test_denoise
from src.test import test_detect_speech
from src.test import test_normalize
from src.test import test_pipeline
from src.test import test_transformation_pipeline
from src.test import test_wav


def run_pytest() -> None:
    """
    Runs all pytest test cases in src/test directory.
    """
    test_dir = Path(__file__).parent / 'src' / 'test'
    pytest.main(["-v", str(test_dir)])


SCRIPTS: dict[str, Callable[..., None]] = {
    "data_analysis": data_analysis.main,
    "from_wav_to_histogram": from_wav_to_histogram.main,
    "generate_rgb_histogram": generate_rgb_histogram.main,
    "regenerate_csv": regenerate_csv.main,
    "spectrogram_script": spectrogram_script.main,
}

TEST_CASES: dict[str, Callable[[], None]] = {
    "wav": test_wav.manual_test,
    "pytest": run_pytest,
    "denoise": test_denoise.manual_test,
    "normalize": test_normalize.manual_test,
    "cut_wav": test_cut_wav.manual_test,
    "cnn": test_cnn.manual_test_cnn,
    "dataset": test_cnn.manual_test_dataset,
    "pipeline": test_pipeline.example_test_run,
    "speech_detection": test_detect_speech.example_test_run,
    "transformation_pipeline": test_transformation_pipeline.example_test_run,
}


def validate_scripts() -> None:
    """
    Validates function signatures in SCRIPTS dictionary.

    Raises:
        TypeError: If function doesn't match expected signature
    """
    for name, func in SCRIPTS.items():
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        if not (len(params) == 0 or (len(params) == 1 and params[0].annotation == list[str])):
            raise TypeError(f"Function '{name}' must take no arguments or a single argument of type list[str].")


def fastapi_main() -> None:
    """
    Starts FastAPI server with the application.
    """
    uvicorn.run("src.frontend.app:app", host="0.0.0.0", port=8000, reload=True)


def display_help() -> None:
    """
    Displays help message showing all available commands and their corresponding
    command-line arguments.
    """
    print("Available commands:")
    print("  train    - Start training")
    print("  validate - Start validation")
    print("  run      - Start running")
    print("  prepare  - Prepare database")
    print("\nOr use command line arguments:")
    print("  -t, --train    - Start training")
    print("  -v, --validate - Start validation")
    print("  -r, --run      - Start running")
    print("  -p, --prepare  - Prepare database")

    print('\n' + '--' * 20 + 'Script help' + '--' * 20)
    print("Usage: python main.py script <script_name> [args...]")
    print("Available scripts:")
    for script_name, func in SCRIPTS.items():
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        if len(params) == 1 and params[0].annotation == list[str]:
            print(f"\t{script_name} <args...>")
        else:
            print(f"\t{script_name}")

    print('\n' + '--' * 20 + 'Test help' + '--' * 20)
    print("Usage: python main.py test <test_name>")
    print("Available tests:")
    for test_name in TEST_CASES:
        print(f"\t{test_name}")


def run_script(script_name: str, args: list[str]) -> None:
    """
    Executes script with given name and arguments.

    Parameters:
        script_name (str): Name of script from SCRIPTS dictionary
        args (list[str]): List of command line arguments for script

    Raises:
        TypeError: If script function has invalid signature
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
        print(f"Script '{script_name}' not found")
        display_help()


def run_test(test_name: str) -> None:
    """
    Executes test with given name.

    Parameters:
        test_name (str): Name of test from TEST_CASES dictionary

    """
    if test_name in TEST_CASES:
        TEST_CASES[test_name]()
    else:
        print(f"Test '{test_name}' not found")
        display_help()


def handle_command(command: str, args: list[str] = None) -> bool:
    """
    Processes command and executes corresponding functionality.

    Parameters:
        command (str): Command to execute (train/validate/run/prepare/script/test)
        args (list[str], optional): Arguments for script/test commands

    Returns:
        bool: True if command was valid and executed successfully
    """
    if command in ('train', '-t', '--train'):
        print("Starting training...")
        train.main()
    elif command in ('validate', '-v', '--validate'):
        print("Starting validation...")
        validate_dataset.main()
    elif command in ('run', '-r', '--run'):
        print("Starting frontend...")
        fastapi_main()
    elif command in ('prepare', '-p', '--prepare'):
        print("Starting db preparation...")
        prepare_dataset.main()
    elif command == 'script' and args:
        print(f"Running script '{args[0]}'...")
        run_script(args[0], args[1:])
    elif command == 'test' and args:
        print(f"Running test '{args[0]}'...")
        run_test(args[0])
    else:
        print("Invalid command!")
        return False
    return True


def parse_arguments() -> None:
    """
    Parses command line arguments and executes corresponding command or starts interactive mode.
    """
    validate_scripts()

    parser = argparse.ArgumentParser(
        description='Process training, validation, running, and database preparation commands.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--train', action='store_true', help='Start training')
    group.add_argument('-v', '--validate', action='store_true', help='Start validation')
    group.add_argument('-r', '--run', action='store_true', help='Start running')
    group.add_argument('-p', '--prepare', action='store_true', help='Prepare database')
    group.add_argument('command', nargs='?', help='Command to execute (train/validate/run/prepare/script/test)')
    group.add_argument('args', nargs=argparse.REMAINDER, help='Additional arguments for scripts')

    args = parser.parse_args()

    if args.train:
        handle_command('train')
    elif args.validate:
        handle_command('validate')
    elif args.run:
        handle_command('run')
    elif args.prepare:
        handle_command('prepare')
    elif args.command:
        if not handle_command(args.command, args.args):
            print("Invalid command!")
            display_help()
    else:
        interactive_mode()


def interactive_mode() -> None:
    """
    Starts interactive command line interface that continuously prompts for commands
    until 'exit' is entered or the program is interrupted.
    """
    print("Entering interactive mode...")
    display_help()

    while True:
        try:
            cmd_input = input("\nEnter command (or 'exit' to quit): ").lower().strip().split()
            if not cmd_input:
                continue

            command = cmd_input[0]
            args = cmd_input[1:] if len(cmd_input) > 1 else None

            if command == 'exit':
                print("Exiting...")
                break

            if not handle_command(command, args):
                print("Invalid command!")
                display_help()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break


def main() -> None:
    """
    Entry Function
    """
    parse_arguments()

if __name__ == "__main__":
    main()
