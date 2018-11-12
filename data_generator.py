import itertools
import math
import random
import threading

import pandas as pd

import numpy as np
import keras as ks

class DataGenerator(ks.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset: pd.DataFrame, dim: tuple, batch_size: int, shuffle: bool):
        self.d_lock = threading.Lock()
        self.spectrograms = dict()
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = shuffle
        self.on_epoch_end()
        self.dataset = dataset
        self.pair_indices = list(itertools.combinations(range(len(dataset)), 2))

    def __len__(self):
        return int(math.floor(len(self.pair_indices) / self.batch_size))

    def __getitem__(self, current_batch):
        'Generate one batch of data'
        # batchsized list of indices of pairs
        batch_indices = self.pair_indices[current_batch * self.batch_size:(current_batch + 1) * self.batch_size]

        # init np array for pairs and corresponding labels
        X = np.empty((self.batch_size, 2 * self.dim[0], self.dim[1]))
        y = np.empty(self.batch_size, dtype=int)

        for i, pair in enumerate(batch_indices):
            row_a = self.dataset.loc[pair[0]]
            row_b = self.dataset.loc[pair[1]]
            X[current_batch][0:self.dim[0]] = load_or_get_spectrogram(row_a['spectrogram_path'])
            X[current_batch][self.dim[0]:2 * self.dim[0]] = load_or_get_spectrogram(row_a['spectrogram_path'])

            if row_a['speaker_id'] == row_b['speaker_id']:
                y[i] = 1
            else:
                y[i] = 0

        return X, y

    def on_epoch_end(self):
        if self.mode == 'shuffle':
            random.shuffle(self.pair_indices)

    def load_or_get_spectrogram(self, filepath):
        """ TODO: fix this ugly hack """
        self.d_lock.acquire()
        if filepath not in self.spectrograms:
            self.spectrograms[filepath] = np.load(filepath)
        self.d_lock.release()

        return self.spectrograms[filepath]
