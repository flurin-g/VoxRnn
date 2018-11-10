from definitions import TRAIN_CONF, WEIGHTS_PATH, NPY_PATH
from vox_rnn_model import train_model
from vox_utils import get_dataset

if __name__ == '__main__':
    configs = TRAIN_CONF
    #data_splits, id_to_label, num_speakers = get_datasets(1)
    train_model(configs, WEIGHTS_PATH)
