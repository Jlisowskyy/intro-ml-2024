"""
Author: Tomasz Mycielski, Jakub Lisowski, 2024

Dataset Generation and Annotation Tool

This script provides functionality for:
1. Generating a dataset from audio files
2. Creating corresponding annotation files
3. Processing audio data in parallel using multiple threads and processes
4. Converting audio files to spectrograms for machine learning tasks
"""

import math
import os
import subprocess
import sys
import threading
from os import walk, path, makedirs

import numpy as np

from src.constants import MODEL_WINDOW_LENGTH, DATABASE_PATH, \
    DATABASE_OUT_NAME, DATABASE_CUT_ITERATOR, CLASSES, \
    DATABASE_ANNOTATIONS_PATH, NORMALIZATION_TYPE, DATABASE_NAME, NUM_THREADS_DB_PREPARE, \
    NUM_PROCESSES_DB_PREPARE
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.wav import FlattenWavIterator, AudioDataIterator


def generate_annotations(dry: bool = False) -> list[str]:
    """
    Creates annotation files for the audio dataset by scanning through the directory structure.

    Args:
        dry (bool): If True, prints annotations instead of writing to file. Used for testing.

    Returns:
        list[str]: List of folder names found in the dataset.

    Notes:
        - Generates a CSV file with columns: folder,file_name,classID
        - Skips '_background_noise_' folder and non-WAV files
        - Handles Mac hidden files by skipping files starting with '.'
    """
    folders = []

    db_path = DATABASE_ANNOTATIONS_PATH if not dry else '/dev/null'
    with open(db_path, 'w', encoding='UTF-8') as f:
        f.write('folder,file_name,classID\n')

        for root, _, files in walk(DATABASE_PATH):
            folder = root.rsplit('/')[-1]

            if folder == '_background_noise_':
                continue

            folders.append(folder)
            new_root = root.replace(DATABASE_NAME, DATABASE_OUT_NAME)

            for file in files:
                if (not file.endswith('.wav')
                        or file.startswith('.')):
                    continue

                class_id = folder if folder in CLASSES else 'unknown'
                data = f'{new_root},{file},{class_id}'

                if dry:
                    print(data)
                else:
                    f.write(data + '\n')

    return folders


class DatabaseGenerator:
    """
    A thread-safe database generator that processes audio files in parallel.

    This class manages multiple worker threads to convert audio files into spectrograms,
    handling concurrent file processing and disk I/O operations safely.

    Attributes:
        _queue (list): Thread-safe queue for storing files to be processed
        _threads (list): List of worker threads
        _should_stop (bool): Flag to signal thread termination
        _lock (threading.Lock): Lock for queue access synchronization
        _sem (threading.Semaphore): Semaphore for queue population control
        _sem_rev (threading.Semaphore): Reverse semaphore for thread pool management
        _file_lock (threading.Lock): Lock for file system operations
    """

    def __init__(self) -> None:
        """
        Initializes the DatabaseGenerator with thread synchronization primitives
        and empty queues/thread pools.
        """
        self._queue = []
        self._threads = []
        self._should_stop = False
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._sem = threading.Semaphore(0)
        self._sem_rev = threading.Semaphore(NUM_THREADS_DB_PREPARE)

    def process(self, target_folder: str) -> None:
        """
        Processes all audio files in the target folder using multiple threads.

        Args:
            target_folder (str): Name of the folder containing audio files to process

        Notes:
            - Creates worker threads based on NUM_THREADS_DB_PREPARE constant
            - Manages thread lifecycle and synchronization
            - Processes WAV files into spectrograms
        """
        for _ in range(NUM_THREADS_DB_PREPARE):
            t = threading.Thread(target=self._worker)
            t.start()
            self._threads.append(t)

        for root, _, files in walk(DATABASE_PATH):
            folder = root.rsplit('/')[-1]

            if folder != target_folder:
                continue

            new_root = root.replace(DATABASE_NAME, DATABASE_OUT_NAME)

            for file in files:
                if (not file.endswith('.wav')
                        or file.startswith('.')
                        or folder == '_background_noise_'):
                    continue

                self._sem_rev.acquire()
                with self._lock:
                    self._queue.append((file, root, new_root))
                self._sem.release()

        self._should_stop = True
        self._sem.release(NUM_THREADS_DB_PREPARE)

        for t in self._threads:
            t.join()

    def _worker(self) -> None:
        """
        Worker thread function that processes files from the shared queue.

        Continuously pulls files from the queue and processes them until
        signaled to stop. Uses semaphores for synchronization and
        thread-safe queue access.
        """
        while True:
            self._sem.acquire()
            with self._lock:
                if self._should_stop:
                    break
                if len(self._queue) == 0:
                    continue
                file, root, new_root = self._queue.pop()
            self._process_file(file, root, new_root)
            self._sem_rev.release()

    def _process_file(self, file: str, root: str, new_root: str) -> None:
        """
        Processes a single audio file into a spectrogram.

        Args:
            file (str): Name of the audio file
            root (str): Source directory path
            new_root (str): Destination directory path

        Notes:
            - Converts audio to fixed-length windows
            - Generates spectrograms using process_audio function
            - Handles padding for shorter audio files
            - Saves output as NumPy arrays
        """
        it = FlattenWavIterator(path.join(root, file), MODEL_WINDOW_LENGTH,
                               DATABASE_CUT_ITERATOR)

        sr = it.get_first_iter().get_frame_rate()
        it = AudioDataIterator(it)
        audio_data = next(iter(it))

        if len(audio_data.audio_signal) < MODEL_WINDOW_LENGTH * sr:
            audio_data.audio_signal = np.pad(audio_data.audio_signal,
                                           (0, MODEL_WINDOW_LENGTH * sr - len(
                                               audio_data.audio_signal)),
                                           constant_values=(0, 0))

        spectrogram = process_audio(audio_data, NORMALIZATION_TYPE)

        if not path.exists(new_root):
            with self._file_lock:
                makedirs(path.join(new_root))

        np.save(path.join(new_root, f'{file[:-4]}.npy'), spectrogram)


def process_func(folder: str) -> None:
    """
    Single-process function for processing a folder of audio files.

    Args:
        folder (str): Name of the folder to process

    Notes:
        - Creates a DatabaseGenerator instance for the folder
        - Used as the target function for multiprocessing
    """
    generator = DatabaseGenerator()
    generator.process(folder)


def chunk_folders(folders: list[str], max_processes: int) -> list[list[str]]:
    """
    Divides folders into chunks for parallel processing.

    Args:
        folders (list[str]): List of folder names to process
        max_processes (int): Maximum number of parallel processes to use

    Returns:
        list[list[str]]: List of folder chunks, each to be processed by one process

    Notes:
        - Ensures even distribution of work across processes
        - Respects maximum process limit
    """
    num_processes = min(max_processes, len(folders))
    chunk_size = math.ceil(len(folders) / num_processes)
    return [folders[i:i + chunk_size] for i in range(0, len(folders), chunk_size)]


def run_process(folders: list[str]) -> None:
    """
    Launches multiple processes to handle folder processing in parallel.

    Args:
        folders (list[str]): List of folders to process

    Notes:
        - Sets up Python environment for subprocesses
        - Handles process creation and monitoring
        - Reports errors from failed processes
    """
    src_dir = path.dirname(path.dirname(path.abspath(__file__)))

    folder_chunks = chunk_folders(folders, NUM_PROCESSES_DB_PREPARE)
    processes = []

    for folder_chunk in folder_chunks:
        cmd = [
            sys.executable,
            "-m",
            f"src.scripts.prepare_dataset",
            '--folders',
            ','.join(folder_chunk)
        ]

        env = os.environ.copy()
        env['PYTHONPATH'] = path.dirname(src_dir) + os.pathsep + env.get('PYTHONPATH', '')

        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            env=env,
            cwd=path.dirname(src_dir)
        )
        processes.append(process)

    for process in processes:
        process.wait()

        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            print(f"Process failed with error: {stderr}", file=sys.stderr)


def main(dry: bool = False) -> None:
    """
    Main entry point for the dataset preparation script.

    Args:
        dry (bool): If True, runs in dry-run mode, printing annotations instead of writing files

    Notes:
        - Coordinates the entire dataset preparation process
        - Generates annotations and launches processing jobs
    """
    folders = generate_annotations(dry)
    print(f'Generated annotations for {len(folders)} folders')

    if not dry:
        run_process(folders)


if __name__ == '__main__':
    # This is ONLY for subprocesses to run the process_func
    if len(sys.argv) > 2 and sys.argv[1] == '--folders':
        folder_list = sys.argv[2].split(',')
        for f in folder_list:
            process_func(f)

            print(f'Processed folder: {f}')
