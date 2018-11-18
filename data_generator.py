import pandas as pd

import numpy as np
import keras as ks

ROWS_PER_LOOP = 2

NUM_OF_PAIRS = 100000


class DataGenerator(ks.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataset: pd.DataFrame, dim: list, batch_size: int, shuffle: bool):
        self.rng = np.random.RandomState(1)
        self.dataset = dataset
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, current_batch):
        'Generate one batch of data'
        df = self.dataset
        # init np array for pairs and corresponding labels
        # TODO: clean up np.concat, rename dim
        batch_start = self.batch_size * current_batch
        batch_end = batch_start + self.batch_size

        X_left = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        X_right = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        y = np.empty(self.batch_size, dtype=np.uint8)

        for i in range(self.batch_size // ROWS_PER_LOOP):
            pos_1 = df.sample(random_state=self.rng).iloc[0]
            pos_2 = df[(df.speaker_id == pos_1.speaker_id) & (df.path != pos_1.path)].sample(random_state=self.rng).iloc[0]

            X_left[i] = np.load(pos_1.spectrogram_path)
            X_right[i] = np.load(pos_2.spectrogram_path)

            # TODO: Fix this ugly hack
            found_neg = False
            while not found_neg:
                neg_1 = df.sample(random_state=self.rng).iloc[0]
                neg_2_candidates = df[(df.Gender == neg_1.Gender) & (df.Nationality == neg_1.Nationality) & (df.speaker_id != neg_1.speaker_id)]
                if len(neg_2_candidates):
                    neg_2 = neg_2_candidates.sample(random_state=self.rng).iloc[0]
                    X_left[i+1] = np.load(neg_1.spectrogram_path)
                    X_right[i+1] = np.load(neg_2.spectrogram_path)
                    found_neg = True

            y[i] = 1
            y[i+1] = 0

        return [[X_left, X_right], y]

    def on_epoch_end(self):
        pass
