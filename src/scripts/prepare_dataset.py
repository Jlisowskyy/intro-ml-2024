"""
Author: Tomasz Mycielski

Helper script for generating a dataset and a relevant annotations file
"""

import math
import os
import subprocess
import sys
import threading
from os import walk, path, makedirs

import numpy as np
from tqdm import tqdm

from src.constants import MODEL_WINDOW_LENGTH, DATABASE_PATH, \
    DATABASE_OUT_NAME, DATABASE_CUT_ITERATOR, CLASSES, \
    DATABASE_ANNOTATIONS_PATH, NORMALIZATION_TYPE, DATABASE_NAME, NUM_THREADS_DB_PREPARE, \
    NUM_PROCESSES_DB_PREPARE
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.wav import FlattenWavIterator, AudioDataIterator


def generate_annotations(dry: bool = False) -> list[str]:
    """
    Generate annotations for the dataset
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
                # Omit annoying hidden mac files
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
    Helper class for generating a dataset and a relevant annotations file
    """

    # ------------------------------
    # Class fields
    # ------------------------------

    _queue: list
    _threads: list
    _should_stop: bool
    _lock: threading.Lock
    _sem: threading.Semaphore
    _sem_rev: threading.Semaphore
    _file_lock: threading.Lock

    # ------------------------------
    # Class init
    # ------------------------------

    def __init__(self) -> None:
        self._queue = []
        self._threads = []
        self._should_stop = False
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._sem = threading.Semaphore(0)
        self._sem_rev = threading.Semaphore(NUM_THREADS_DB_PREPARE)

    # ------------------------------
    # Class methods
    # ------------------------------

    def process(self, target_folder: str) -> None:
        """
        Process the dataset
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
                # Omit annoying hidden mac files
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

    # ------------------------------
    # Protected methods
    # ------------------------------

    def _worker(self) -> None:
        """
        Worker method
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
        Process a single file
        """

        it = FlattenWavIterator(path.join(root, file), MODEL_WINDOW_LENGTH,
                                DATABASE_CUT_ITERATOR)

        sr = it.get_first_iter().get_frame_rate()
        it = AudioDataIterator(it)
        audio_data = next(iter(it))

        # Pad not full files
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
    Function for each running process
    """

    generator = DatabaseGenerator()
    generator.process(folder)


def chunk_folders(folders: list[str], max_processes: int) -> list[list[str]]:
    """
    Chunk the folders
    """

    num_processes = min(max_processes, len(folders))
    chunk_size = math.ceil(len(folders) / num_processes)
    return [folders[i:i + chunk_size] for i in range(0, len(folders), chunk_size)]


def run_process(folders: list[str]) -> None:
    """
    Run the processes
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
            stdout=subprocess.PIPE,
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
    Main method
    """
    folders = generate_annotations(dry)
    run_process(folders)


if __name__ == '__main__':
    # This is ONLY for subprocesses to run the process_func
    if len(sys.argv) > 2 and sys.argv[1] == '--folders':
        folder_list = sys.argv[2].split(',')
        for f in folder_list:
            process_func(f)
