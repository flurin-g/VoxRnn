from definitions import TRAIN_CONF, WEIGHTS_PATH
from vox_rnn_model import train_model

if __name__ == '__main__':
    configs = TRAIN_CONF
    train_model(configs, WEIGHTS_PATH)
