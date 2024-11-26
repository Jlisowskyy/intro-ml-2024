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
import traceback
from pathlib import Path
from collections.abc import Callable

import pytest
import uvicorn
from colorama import Fore, Style, init

from src.cnn import train
from src.scripts import data_analysis
from src.scripts import generate_rgb_histogram
from src.scripts import prepare_dataset
from src.scripts import regenerate_csv
from src.scripts import spectrogram_script
from src.scripts import validate_dataset
from src.test import test_classify, test_fit_to_window
from src.test import test_cnn
from src.test import test_cut_wav
from src.test import test_denoise
from src.test import test_detect_speech
from src.test import test_normalize
from src.test import test_transformation_pipeline
from src.test import test_wav

# Initialize colorama
init()

def print_error(message: str) -> None:
    """
    Prints error message in red color.

    Parameters:
        message (str): Error message to display
    """
    print(f"{Fore.RED}{message}{Style.RESET_ALL}")


def print_warning(message: str) -> None:
    """
    Prints warning message in yellow color.

    Parameters:
        message (str): Warning message to display
    """
    print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")


def print_success(message: str) -> None:
    """
    Prints success message in green color.

    Parameters:
        message (str): Success message to display
    """
    print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")


def run_pytest() -> None:
    """
    Runs all pytest test cases in src/test directory.
    """
    test_dir = Path(__file__).parent / 'src' / 'test'
    pytest.main(["-v", str(test_dir)])


SCRIPTS: dict[str, Callable[..., None]] = {
    "data_analysis": data_analysis.main,
    "generate_rgb_histogram": generate_rgb_histogram.main,
    "regenerate_csv": regenerate_csv.main,
    "spectrogram_script": spectrogram_script.main,
}

TEST_CASES: dict[str, Callable[[], None]] = {
    "wav": test_wav.manual_test,
    "pytest": run_pytest,
    "denoise": test_denoise.denoise_test_manual,
    "normalize": test_normalize.manual_test,
    "cut_wav": test_cut_wav.manual_test,
    "cnn": test_cnn.manual_test_cnn,
    "dataset": test_cnn.manual_test_dataset,
    "silence_removal": test_detect_speech.silence_removal_test,
    "transformation_pipeline": test_transformation_pipeline.example_test_run,
    "test_classify": test_classify.example_test_run,
    "fit_to_window": test_fit_to_window.fit_to_window_test
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
            raise TypeError(
                f"Function '{name}' must take no arguments or a single argument of type list[str].")


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
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}Available commands:{Style.RESET_ALL}")
    print("  train    - Start training")
    print("  validate - Start validation")
    print("  run      - Start running")
    print("  prepare  - Prepare database")
    print(f"\n{Fore.CYAN}Command line usage:{Style.RESET_ALL}")
    print("  python main.py [-h] [-t | -v | -r | -p | command [args ...]]")
    print("  -h, --help    - Show this help message")
    print("  -t, --train   - Start training")
    print("  -v, --validate- Start validation")
    print("  -r, --run     - Start running")
    print("  -p, --prepare - Prepare database")

    print('\n' + '==' * 17 + f' {Fore.CYAN}Script help{Style.RESET_ALL} ' + '==' * 17)
    print("Usage: python main.py script <script_name> [args...]")
    print("Scripts path: ./src/helper_scripts/")
    print("Available scripts:")
    for script_name, func in SCRIPTS.items():
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        if len(params) == 1 and params[0].annotation == list[str]:
            print(f"\t{script_name} <args...>")
        else:
            print(f"\t{script_name}")

    print('\n' + '==' * 18 + f' {Fore.CYAN}Test help{Style.RESET_ALL} ' + '==' * 17)
    print("Usage: python main.py test <test_name>")
    print("Tests path: ./src/test/")
    print("Available tests:")
    for test_name in TEST_CASES:
        print(f"\t{test_name}")
    print("=" * 80 + "\n")


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
        print_error(f"Script '{script_name}' not found")
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
        print_error(f"Test '{test_name}' not found")
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
    try:
        if command in ('train', '-t', '--train'):
            print_success("Starting training...")
            train.main()
        elif command in ('validate', '-v', '--validate'):
            print_success("Starting validation...")
            validate_dataset.main()
        elif command in ('noise'):
            print_success("Starting noise preparation...")
            prepare_noises.main()
        elif command in ('run', '-r', '--run'):
            print_success("Starting frontend...")
            fastapi_main()
        elif command in ('prepare', '-p', '--prepare'):
            print_success("Starting db preparation...")
            prepare_dataset.main('dry' in args)
        elif command == 'script' and args:
            print_success(f"Running script '{args[0]}'...")
            run_script(args[0], args[1:])
        elif command == 'test':
            if not args:
                print_error("No test name provided!")
                display_help()
                return False
            print_success(f"Running test '{args[0]}'...")
            run_test(args[0])
        else:
            print_error("Invalid command!")
            return False
        return True
    # pylint: disable=broad-except
    except Exception:
        print_error(f"Error executing command: {traceback.format_exc()}")
        return False


def parse_arguments() -> None:
    """
    Parses command line arguments and executes corresponding command or starts interactive mode.
    """
    validate_scripts()

    parser = argparse.ArgumentParser(
        description='Process training, validation, running, and database preparation commands.')

    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('-t', '--train', action='store_true', help='Start training')
    main_group.add_argument('-v', '--validate', action='store_true', help='Start validation')
    main_group.add_argument('-r', '--run', action='store_true', help='Start running')
    main_group.add_argument('-p', '--prepare', action='store_true', help='Prepare database')

    parser.add_argument('command', nargs='?',
                        help='Command to execute (train/validate/run/prepare/script/test)')
    parser.add_argument('args', nargs='*', help='Additional arguments for scripts or tests')

    args = parser.parse_args()
    cmd = None
    if args.train:
        cmd = 'train'
    elif args.validate:
        cmd = 'validate'
    elif args.run:
        cmd = 'run'
    elif args.prepare:
        cmd = 'prepare'
    elif args.command:
        if not handle_command(args.command, args.args):
            display_help()
    else:
        interactive_mode()
    if cmd is not None:
        handle_command(cmd, args.args) # TODO: Add subflags


def interactive_mode() -> None:
    """
    Starts interactive command line interface that continuously prompts for commands
    until 'exit' is entered or the program is interrupted.
    """
    print_success("Entering interactive mode...")
    display_help()

    while True:
        try:
            cmd_input = input(f"\n{Fore.CYAN}Enter command"
                              f" (or 'exit' to quit):{Style.RESET_ALL} ").lower().strip().split()
            if not cmd_input:
                continue

            command = cmd_input[0]
            args = cmd_input[1:] if len(cmd_input) > 1 else None

            if command == 'exit':
                print_success("Exiting...")
                break

            if not handle_command(command, args):
                print_error("Invalid command!")
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
