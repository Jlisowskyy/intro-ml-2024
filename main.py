"""
Author: Jakub Lisowski, 2024

This script is a simple command line interface for training, validation, running the model,
and database preparation. It provides both command-line argument parsing and an interactive mode
for executing various operations.
The available commands are:
- train: Start the model training process
- validate: Validate the dataset
- run: Start the FastAPI frontend
- prepare: Prepare the database
"""

import argparse

import uvicorn

from src.cnn import train
from src.helper_scripts import validate_dataset
from src.helper_scripts import prepare_dataset


def fastapi_main() -> None:
    """
    Start the FastAPI server with the application.
    """
    uvicorn.run("src.frontend.app:app", host="0.0.0.0", port=8000, reload=True)


def display_parse_help() -> None:
    """
    Display the help message showing all available commands and their corresponding
    command-line arguments.

    This function prints both the interactive mode commands and their equivalent
    command-line arguments for user reference.
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


def handle_command(command: str) -> bool:
    """
    Execute the appropriate function based on the given command.

    :param command: The command to execute.

    :returns: True if the command was recognized and executed successfully,
              False if the command was not recognized.
    """
    if command == 'train' or command == '-t' or command == '--train':
        print("Starting training...")
        train.main()
    elif command == 'validate' or command == '-v' or command == '--validate':
        print("Starting validation...")
        validate_dataset.main()
    elif command == 'run' or command == '-r' or command == '--run':
        print("Starting frontend...")
        fastapi_main()
    elif command == 'prepare' or command == '-p' or command == '--prepare':
        print("Starting db preparation...")
        prepare_dataset.main()
    else:
        return False
    return True


def parse_arguments():
    """
    Parse command-line arguments and execute the corresponding command.

    This function sets up the argument parser with mutually exclusive options
    for different operations. If no arguments are provided, it falls back to
    interactive mode.
    """
    parser = argparse.ArgumentParser(
        description='Process training, validation, running, and database preparation commands.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--train', action='store_true', help='Start training')
    group.add_argument('-v', '--validate', action='store_true', help='Start validation')
    group.add_argument('-r', '--run', action='store_true', help='Start running')
    group.add_argument('-p', '--prepare', action='store_true', help='Prepare database')

    args = parser.parse_args()

    if args.train:
        handle_command('train')
    elif args.validate:
        handle_command('validate')
    elif args.run:
        handle_command('run')
    elif args.prepare:
        handle_command('prepare')
    else:
        interactive_mode()


def interactive_mode():
    """
    Start an interactive command-line interface.

    This function runs a loop that continuously prompts the user for commands
    until 'exit' is entered or the program is interrupted. Invalid commands
    will display the help message.
    """
    print("Entering interactive mode...")
    display_parse_help()

    while True:
        try:
            command = input("\nEnter command (or 'exit' to quit): ").lower().strip()

            if command == 'exit':
                print("Exiting...")
                break

            if not handle_command(command):
                print("Invalid command!")
                display_parse_help()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break


if __name__ == "__main__":
    parse_arguments()