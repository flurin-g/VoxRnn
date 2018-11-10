import math
import os.path
import numpy as np
import keras as ks

from definitions import NPY_PATH


class DataGenerator(ks.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs: dict, labels: dict, batch_size: int,
                 dim: tuple, n_classes: int, npy_path: os.path, shuffle: bool):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.mode = shuffle
        self.npy_path = npy_path
        self.n_classes = n_classes
        self.mode = shuffle
        self.on_epoch_end()

        self.unique_ids = np.array(set(list_IDs.values()))
        self.id_to_label = {v: k for k, v in labels.items()}

        # muss key: values sein um von id
        self.id_indices = [current_id: np.where(id_to_label == i) for current_id in self.unique_ids]
        self.indexes = np.arange(len(self.list_IDs))

        self.indexes =

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(math.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        for id in self.unique_ids:




        if self.mode == 'shuffle':
            np.random.shuffle(self.indexes)


