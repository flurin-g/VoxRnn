import itertools
import math
import os
import random

import pandas as pd

import numpy as np
import keras as ks

from definitions import NPY_PATH


class DataGenerator(ks.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset: pd.DataFrame, dim: tuple, batch_size: int, shuffle: bool):
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = shuffle
        self.on_epoch_end()
        self.dataset = pd.read_csv('/Users/flurin/repos/hs18/pa/VoxRnn/Data/same_speakers_dev.csv',
                                   nrows=100)
        self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return int(math.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, current_batch):
        'Generate one batch of data'
        # init np array for pairs and corresponding labels
        # TODO: clean up np.concat, rename dim
        X_left = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        X_right = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        y = np.empty(self.batch_size, dtype=int)

        for i in range(self.batch_size):
            X_left[i] = np.load(self.file_name_temp(self.dataset.loc[self.indices[i], 'utterance_a']))
            X_right[i] = np.load(self.file_name_temp(self.dataset.loc[self.indices[i], 'utterance_b']))

            if self.dataset.loc[i, 'y_label']:
                y[i] = 1
            else:
                y[i] = 0

        return [[X_left, X_right], y]

    def on_epoch_end(self):
        if self.mode == 'shuffle':
            random.shuffle(self.indices)

    @staticmethod
    def file_name_temp(file_name_raw: str):
        return os.path.join(NPY_PATH, file_name_raw.replace('/', '-').replace('.', '-')) + ".npy"
