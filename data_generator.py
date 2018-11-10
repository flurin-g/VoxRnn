"""
Custom Data-Generator for Keras, allows to load larger-than-memory data-sets to
be streamed directly from disc.

Code is based on the suggestions of Shervine Amidi from Stanford, see:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import math
import os.path
import numpy as np
import keras as ks

from definitions import NPY_PATH


class DataGenerator(ks.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs: dict, labels: dict, batch_size: int,
                 dim: tuple, n_classes: int, npy_path: os.path, mode: str):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.mode = mode
        self.npy_path = npy_path
        self.n_classes = n_classes
        self.mode = mode
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(math.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.mode == 'shuffle':
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        #itertools.combinations
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(os.path.join(NPY_PATH, ID))

            # Store class
            y[i] = self.labels[ID]

        return X, ks.utils.to_categorical(y, num_classes=self.n_classes)
