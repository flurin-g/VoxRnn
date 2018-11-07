import os
import sys


import yaml
import pandas as pd
import numpy as np
import librosa as lr

from definitions import GLOBAL_CONF

TRAIN = 0  # in orig file + 1, but because lists are zero based...
DEV = 1
TEST = 2
NUM_DATA_SETS = 2


def get_path(name: str) -> str:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    name = os.path.join(current_directory, name)
    return name


def convert_to_id(path_name: str) -> str:
    return path_name.replace('/', '-').replace('.', '-') + '.npy'


def write_to_disk(mel_spectrogram: np.ndarray, dest_dir: str, file_id: str):
    path = os.path.join(dest_dir, file_id)
    np.save(path, mel_spectrogram, allow_pickle=False)


def load_rows(name: str) -> pd.DataFrame:
    path = get_path(name)
    return pd.read_csv(path, sep=' ', names=['data_set', 'path'], header=None)


def select_rows(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    # for mode train, load only data_set 1 and 2 (train and dev)
    if mode == 'all':
        rows = rows
    if mode == 'train-dev':
        rows = rows[rows['data_set'] != TEST + 1]

    return rows


def split_rows(row) -> tuple:
    """
    :param row:
    :return:  data_set = train/dev/test-set, speaker_label = path[2:7], file_id = path
    """
    return (row['data_set'] - 1), row['path'][2:7], row['path']


def create_spectrogram(file_id: str, offset_range: list,
                       sampling_rate: int, sample_length: float,
                       fft_window: int, hop_length: int, channels: int) -> np.ndarray:
    offset = np.random.uniform(offset_range[0], offset_range[1])

    if channels == 1:
        mono = True

    audio_range, _ = lr.load(path=file_id,
                             sr=sampling_rate,
                             mono=mono,
                             offset=offset,
                             duration=sample_length)

    mel_spectrogram = lr.feature.melspectrogram(y=audio_range,
                                                sr=sampling_rate,
                                                n_fft=fft_window,
                                                hop_length=hop_length)

    # Compress spectrogram to weighted db-scale
    return dynamic_range_compression(mel_spectrogram)


def dynamic_range_compression(spectrogram):
    return np.log10(1 + np.multiply(10000, spectrogram))


def get_datasets(channels: int) -> tuple:
    """
    :return: tuple containing train and test sets as ndarray
    """
    configs = GLOBAL_CONF

    # directory of voxCeleb dev wave files
    vox_dev_wav = get_path(configs['files']['vox_dev_wav'])
    # directory to store numpy binaries
    npy_dir = get_path(configs['files']['numpy'])

    data_splits = dict()
    data_splits['train'] = list()
    data_splits['dev'] = list()

    id_to_label = dict()

    rows = load_rows(configs['files']['vox_celeb_splits'])

    # select subset according to vox1_meta
    rows = select_rows(rows, configs['dataset']['split'])

    mel_config = configs['spectrogram']
    for _, row in rows.iterrows():
        data_set, speaker_label, path_name = split_rows(row)

        mel_spectrogram = create_spectrogram(vox_dev_wav + "/" + path_name,
                                             mel_config['offset_range'],
                                             mel_config['sampling_rate'],
                                             mel_config['sample_length'],
                                             mel_config['fft_window'],
                                             mel_config['hop_length'],
                                             channels)

        file_id: str = convert_to_id(path_name)

        write_to_disk(mel_spectrogram, npy_dir, file_id)

        # add label to train/dev-set _y
        if data_set == TRAIN:
            data_splits['train'].append(file_id)
        elif data_set == DEV:
            data_splits['dev'].append(file_id)

        id_to_label[file_id] = speaker_label

    num_speakers = len(id_to_label)

    return data_splits, id_to_label, num_speakers


if __name__ == "__main__":
    get_datasets(1)
    print('success')
