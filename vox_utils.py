import os
import sys

import yaml
from pandas import read_fwf
from pandas import DataFrame
from numpy import ndarray, random, multiply, log10, save
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


def get_path(name: str) -> str:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    name = os.path.join(current_directory, name)
    return name


def convert_to_id(path_name: str) -> str:
    return path_name.replace('/', '-').replace('.', '-') + '.npy'


def write_to_disk(mel_spectrogram: ndarray, dest_dir: str, file_id: str):
    path = os.path.join(dest_dir, file_id)
    save(path, mel_spectrogram, allow_pickle=False)


def load_rows(name: str) -> DataFrame:
    path = get_path(name)
    return read_fwf(path, names=['data_set', 'path'])


def select_rows(rows: DataFrame, mode: str) -> DataFrame:
    # for mode train, load only data_set 1 and 2 (train and dev)
    if mode == all:
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
                       fft_window: int, hop_length: int, channels: int) -> ndarray:
    offset = random.uniform(offset_range[0], offset_range[1])

    if channels == 1:
        mono = True

    audio_range, _ = load(path=file_id,
                          sr=sampling_rate,
                          mono=mono,
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


def get_datasets() -> tuple:
    """
    Generates train_x, train_y, dev_x, dev_y, test_x, test_y from a given dataset
    :return: tuple containing train and test sets as ndarray
    """
    configs = load_config()

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
                                             configs['channels'])

        file_id: str = convert_to_id(path_name)

        write_to_disk(mel_spectrogram, npy_dir, file_id)

        # add label to train/dev-set _y
        if data_set == TRAIN:
            data_splits['train'].append(file_id)
        elif data_set == DEV:
            data_splits['dev'].append(file_id)

        id_to_label[file_id] = speaker_label

    num_speakers = 0 # ToDo: implement (use unique speaker ids to calculate
    return data_splits, id_to_label, num_speakers


if __name__ == "__main__":
    get_datasets()
    print('success')
