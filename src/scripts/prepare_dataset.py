"""
Author: Tomasz Mycielski

Helper script for generating a dataset and a relevant annotations file
"""

import threading
from os import walk, path, makedirs
from typing import TextIO

import numpy as np
from tqdm import tqdm

from src.constants import MODEL_WINDOW_LENGTH, DATABASE_PATH, \
    DATABASE_OUT_NAME, DATABASE_CUT_ITERATOR, CLASSES, \
    DATABASE_ANNOTATIONS_PATH, NORMALIZATION_TYPE, DATABASE_NAME, NUM_THREADS_DB_PREPARE
from src.pipeline.base_preprocessing_pipeline import process_audio
from src.pipeline.wav import FlattenWavIterator, AudioDataIterator


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

    _file: TextIO
    _file_lock: threading.Lock
    _dry: bool

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
        self._dry = False

    # ------------------------------
    # Class methods
    # ------------------------------

    def process(self, dry: bool) -> None:
        """
        Process the dataset
        """

        self._dry = dry
        for _ in range(NUM_THREADS_DB_PREPARE):
            t = threading.Thread(target=self._worker)
            t.start()
            self._threads.append(t)

        # TODO: add NUL support for the windows users
        db_path = DATABASE_ANNOTATIONS_PATH if not dry else '/dev/null'
        with open(db_path, 'w', encoding='UTF-8') as f:
            f.write('folder,file_name,classID\n')
            self._file = f

            for root, _, files in walk(DATABASE_PATH):
                folder = root.rsplit('/')[-1]
                new_root = root.replace(DATABASE_NAME, DATABASE_OUT_NAME)

                for file in tqdm(files, colour='magenta'):
                    # Omit annoying hidden mac files
                    if (not file.endswith('.wav')
                            or file.startswith('.')
                            or folder == '_background_noise_'):
                        continue

                    self._sem_rev.acquire()
                    with self._lock:
                        self._queue.append((file, folder, root, new_root))
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
                file, folder, root, new_root = self._queue.pop()
            self._process_file(file, folder, root, new_root)
            self._sem_rev.release()

    def _process_file(self, file: str, folder: str, root: str, new_root: str) -> None:
        """
        Process a single file
        """

        it = FlattenWavIterator(path.join(root, file), MODEL_WINDOW_LENGTH,
                                DATABASE_CUT_ITERATOR)

        sr = it.get_first_iter().get_frame_rate()
        it = AudioDataIterator(it)

        for audio_data in it:
            # Pad not full files
            if len(audio_data.audio_signal) < MODEL_WINDOW_LENGTH * sr:
                audio_data.audio_signal = np.pad(audio_data.audio_signal,
                                                 (0, MODEL_WINDOW_LENGTH * sr - len(
                                                     audio_data.audio_signal)),
                                                 constant_values=(0, 0))

            spectrogram = process_audio(audio_data, NORMALIZATION_TYPE)

            class_id = folder if folder in CLASSES else 'unknown'
            data = f'{new_root},{file},{class_id}'

            if self._dry:
                print(data)
            else:
                if not path.exists(new_root):
                    with self._file_lock:
                        makedirs(path.join(new_root))

                np.save(path.join(new_root, f'{file[:-4]}.npy'), spectrogram)

                with self._file_lock:
                    self._file.write(data + '\n')
            break  # Apologies to future me for the legacy code, it was easier to adapt it


def main(dry: bool = False) -> None:
    """
    Main method
    """
    generator = DatabaseGenerator()
    generator.process(dry)
