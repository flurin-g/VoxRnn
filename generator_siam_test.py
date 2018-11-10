import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist

class DataGenerator(object):
    """docstring for DataGenerator"""
    def __init__(self, batch_sz=64):
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_train /= 255
        X_test /= 255

        # create training+test positive and negative pairs
        digit_indices = [np.where(y_train == i)[0] for i in range(10)]
        self.tr_pairs, self.tr_y = self.create_pairs(X_train, digit_indices)

        digit_indices = [np.where(y_test == i)[0] for i in range(10)]
        self.te_pairs, self.te_y = self.create_pairs(X_test, digit_indices)

        self.tr_pairs_0 = self.tr_pairs[:, 0]
        self.tr_pairs_1 = self.tr_pairs[:, 1]
        self.te_pairs_0 = self.te_pairs[:, 0]
        self.te_pairs_1 = self.te_pairs[:, 1]

        self.batch_sz = batch_sz
        self.samples_per_train  = (self.tr_pairs.shape[0]/self.batch_sz)*self.batch_sz
        self.samples_per_val    = (self.te_pairs.shape[0]/self.batch_sz)*self.batch_sz


        self.cur_train_index=0
        self.cur_val_index=0

    def create_pairs(self, x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_sz
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index=0
            yield ([    self.tr_pairs_0[self.cur_train_index:self.cur_train_index+self.batch_sz],
                        self.tr_pairs_1[self.cur_train_index:self.cur_train_index+self.batch_sz]
                    ],
                    self.tr_y[self.cur_train_index:self.cur_train_index+self.batch_sz]
                )

    def next_val(self):
        while 1:
            self.cur_val_index += self.batch_sz
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index=0
            yield ([    self.te_pairs_0[self.cur_val_index:self.cur_val_index+self.batch_sz],
                        self.te_pairs_1[self.cur_val_index:self.cur_val_index+self.batch_sz]
                    ],
                    self.te_y[self.cur_val_index:self.cur_val_index+self.batch_sz]
                )

hey = DataGenerator()