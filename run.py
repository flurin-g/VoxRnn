import os

from definitions import NPY_PATH, COF
from vox_rnn_model import train_model
from utils import create_all_spectrograms, create_pairs

if __name__ == '__main__':
    if not os.listdir(NPY_PATH):
        create_all_spectrograms()
        print('spectrograms created')


        def train_model(configs: dict, weights_path: str)
