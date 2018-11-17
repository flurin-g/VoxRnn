import os

from definitions import NPY_PATH, TRAIN_CONF, WEIGHTS_PATH
from vox_rnn_model import train_model
from utils import create_all_spectrograms, create_pairs

if __name__ == '__main__':
    if not os.listdir(NPY_PATH):
        create_all_spectrograms()
        print('spectrograms created')

    train_model()


