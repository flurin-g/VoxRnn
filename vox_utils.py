import os
import sys

import yaml
from pandas import read_fwf
from pandas import DataFrame
from numpy import ndarray, random, multiply, log10
from librosa import load
from librosa.feature import melspectrogram

TRAIN = 0  # in orig file + 1, but because lists are zero based...
DEV = 1
TEST = 2
NUM_DATA_SETS = 2


def load_config() -> dict:
    with open("global_configs.yaml", 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print("error parsing file")
            sys.exit(1)


def load_rows() -> DataFrame:
    filename = os.path.realpath('Data/iden_split.txt')
    return read_fwf(filename, names=['data_set', 'path'])


def select_rows(rows: DataFrame, mode: str) -> DataFrame:
    # for mode train, load only data_set 1 and 2 (train and dev)
    if mode == all:
        rows = rows

    return rows


def split_rows(row) -> tuple:
    """
    :param row:
    :return:  data_set = train/dev/test-set, speaker_label = path[2:7], file_id = path
    """
    return (row['data_set'] - 1), row['path'][2:7], row['path']


def create_spectrogram(file_id: str, offset_range: list, sampling_rate: int,
                       sample_length: float, fft_window: int, hop_length: int) -> ndarray:
    offset = random.uniform(offset_range[0], offset_range[1])

    audio_range, _ = load(path=file_id,
                          sr=sampling_rate,
                          mono=True,
                          offset=offset,
                          duration=sample_length)

    mel_spectrogram = melspectrogram(y=audio_range,
                                     sr=sampling_rate,
                                     n_fft=fft_window,
                                     hop_length=hop_length)

    # Compress spectrogram to weighted db-scale
    return dynamic_range_compression(mel_spectrogram)


def dynamic_range_compression(spectrogram):
    return log10(1 + multiply(10000, spectrogram))


def convert_to_id(path_name: str) -> str:
    return path_name.replace('/', '-')


def write_to_disk(mel_spectrogram: ndarray, file_id: str):
    # ToDo: write to disk as ndarray
    pass


def get_datasets() -> tuple:
    """
    Generates train_x, train_y, dev_x, dev_y, test_x, test_y from a given dataset
    :return: tuple containing train and test sets as ndarray
    """
    configs = load_config()

    data_splits = dict()
    data_splits['train'] = list()
    data_splits['dev'] = list()

    id_to_label = dict()

    rows = load_rows()

    # select subset according to vox1_meta
    rows = select_rows(rows, configs['dataset']['split'])

    mel_config = configs['spectrogram']
    for _, row in rows.iterrows():
        data_set, speaker_label, path_name = split_rows(row)

        mel_spectrogram = create_spectrogram(mel_config['path_name'],
                                             mel_config['offset_range'],
                                             mel_config['sampling_rate'],
                                             mel_config['sample_length'],
                                             mel_config['fft_window'],
                                             mel_config['hop_length'])

        file_id = convert_to_id(path_name)

        write_to_disk(mel_spectrogram, file_id)

        # add label to train/dev-set _y
        if data_set == TRAIN:
            data_splits['train'].append(file_id)
        elif data_set == DEV:
            data_splits['dev'].append(file_id)

        id_to_label[file_id] = speaker_label


if __name__ == "__main__":
    get_datasets()
    print('success')
