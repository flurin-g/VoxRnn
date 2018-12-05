import pandas as pd

import numpy as np
import keras as ks
from keras.utils import Sequence

ROWS_PER_LOOP = 2

NUM_OF_PAIRS = 100000


class DataGenerator(Sequence):
    """Generates data for Keras"""

    @staticmethod
    def generate_batch(df: pd.DataFrame, batch_size: int, dim_0: int, dim_1: int, random_state: np.random.RandomState):
        X_left = np.empty((batch_size, dim_0, dim_1))
        X_right = np.empty((batch_size, dim_0, dim_1))
        y = np.empty(batch_size, dtype=np.uint8)

        for i in range(0, batch_size, 2):
            pos_1 = df.sample(random_state=random_state).iloc[0]
            pos_2 = (df[(df.speaker_id == pos_1.speaker_id) & (df.path != pos_1.path)]
                .sample(random_state=random_state)
                .iloc[0])

            X_left[i] = np.load(pos_1.spectrogram_path)
            X_right[i] = np.load(pos_2.spectrogram_path)

            found_neg = False
            while not found_neg:
                neg_1 = df.sample(random_state=random_state).iloc[0]
                neg_2_candidates = df[(df.Gender == neg_1.Gender) & (df.Nationality == neg_1.Nationality) & (
                        df.speaker_id != neg_1.speaker_id)]
                if not neg_2_candidates.empty:
                    neg_2 = neg_2_candidates.sample(random_state=random_state).iloc[0]
                    X_left[i + 1] = np.load(neg_1.spectrogram_path)
                    X_right[i + 1] = np.load(neg_2.spectrogram_path)
                    found_neg = True

            y[i] = 1
            y[i + 1] = 0
        return [[X_left, X_right], y]

    def __init__(self, dataset: pd.DataFrame, dim: list, batch_size: int):
        self.rng = np.random.RandomState(1)
        self.dataset = dataset
        self.dim = dim
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, current_batch):
        'Generate one batch of data'
        return self.generate_batch(self.dataset, self.batch_size, self.dim[0], self.dim[1], self.rng)
