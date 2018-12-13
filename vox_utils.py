import os

import librosa as lr
import numpy as np
import pandas as pd

from definitions import GLOBAL_CONF

TRAIN = 1
DEV = 2
TEST = 3


def get_path(name: str) -> str:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    name = os.path.join(current_directory, name)
    return name


def get_wav_path(split, path):
    files = GLOBAL_CONF['files']
    if split == TEST:
        return get_path(os.path.join(files['vox_test_wav'], path))
    else:
        return get_path(os.path.join(files['vox_dev_wav'], path))


def persist_spectrogram(mel_spectrogram: np.ndarray, wav_path: str):
    np.save(wav_path, mel_spectrogram, allow_pickle=False)


def create_spectrogram(file_path: os.path, offset: float,
                       sampling_rate: int, sample_length: float,
                       fft_window: int, hop_length: int) -> np.ndarray:
    audio_range, _ = lr.load(path=file_path,
                             sr=sampling_rate,
                             mono=True,
                             offset=offset,
                             duration=sample_length)

    # librosa uses centered frames, the result will always be +1 frame, therefore subtract 1 frame
    audio_range = audio_range[:-1]
    mel_spectrogram = lr.feature.melspectrogram(y=audio_range,
                                                sr=sampling_rate,
                                                n_fft=fft_window,
                                                hop_length=hop_length)

    # Compress spectrogram to weighted db-scale
    return np.rot90(dynamic_range_compression(mel_spectrogram))


def dynamic_range_compression(spectrogram):
    return np.log10(1 + np.multiply(10000, spectrogram))


def get_dataset(build_spectrograms=False) -> pd.DataFrame:
    """
    :return: DataFrame containing dataset with metadata and filepaths
    """
    configs = GLOBAL_CONF

    meta = pd.read_csv(
        configs['files']['vox_celeb_meta'],
        sep='\t',
        index_col=0
    )

    meta.index.name = 'speaker_id'

    splits = pd.read_csv(
        configs['files']['vox_celeb_splits'],
        sep=' ',
        names=['split', 'path'],
        header=None
    )

    splits['speaker_id'] = splits['path'].apply(lambda p: p.split('/')[0])
    splits['wav_path'] = splits.apply(
        lambda r: get_wav_path(r['split'], r['path']),
        axis='columns'
    )

    dataset = pd.merge(splits, meta, how='left', on='speaker_id', validate="m:1")

    dataset['spectrogram_path'] = dataset['wav_path'].apply(lambda p: p + '.npy')

    mel_config = configs['spectrogram']
    if build_spectrograms:
        for _, row in dataset.iterrows():
            wav_path = row['wav_path']
            spectrogram_path = row['spectrogram_path']
            if not os.path.exists(spectrogram_path):
                mel_spectrogram = create_spectrogram(wav_path,
                                                     mel_config['offset'],
                                                     mel_config['sampling_rate'],
                                                     mel_config['sample_length'],
                                                     mel_config['fft_window'],
                                                     mel_config['hop_length'])

                persist_spectrogram(mel_spectrogram, spectrogram_path)

    return dataset


def get_train_set(build_spectrograms=False) -> pd.DataFrame:
    """
    :return: DataFrame containing train data with metadata and filepaths
    """

    return get_all_sets(build_spectrograms)[0]


def get_dev_set(build_spectrograms=False) -> pd.DataFrame:
    """
    :return: DataFrame containing dev data with metadata and filepaths
    """
    return get_all_sets(build_spectrograms)[1]


def get_test_set(build_spectrograms=False) -> pd.DataFrame:
    """
    :return: DataFrame containing test data with metadata and filepaths
    """

    return get_all_sets(build_spectrograms)[2]


def get_all_sets(build_spectrograms=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :return: DataFrame containing all datasets with metadata and filepaths
    """
    df = get_dataset(build_spectrograms)

    train_set = df[df.split == TRAIN]
    dev_set = df[df.split == DEV]
    test_set = df[df.split == TEST]

    return train_set, dev_set, test_set
