import os

import librosa as lr
import numpy as np
import pandas as pd

from definitions import GLOBAL_CONF, NPY_PATH

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


def convert_to_spectrogram_filename(path_name: str) -> str:
    return path_name.replace('/', '-').replace('.', '-') + '.npy'


def persist_spectrogram(mel_spectrogram: np.ndarray, spectrogram_path: str):
    np.save(spectrogram_path, mel_spectrogram, allow_pickle=False)


def create_spectrogram(file_id: str, offset_range: list,
                       sampling_rate: int, sample_length: float,
                       fft_window: int, hop_length: int, channels: int = 1) -> np.ndarray:
    offset = np.random.uniform(offset_range[0], offset_range[1])

    if channels == 1:
        mono = True
    else:
        mono = False

    audio_range, _ = lr.load(path=file_id,
                             sr=sampling_rate,
                             mono=mono,
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


def get_dataset() -> pd.DataFrame:
    """
    :return: DataFrame containing dataset with metadata and filepaths
    """
    configs = GLOBAL_CONF

    meta = pd.read_csv(
        configs['files']['vox_celeb_meta'],
        sep='\t',
        index_col=0
    )

    splits = pd.read_csv(
        configs['files']['vox_celeb_splits'],
        sep=' ',
        names=['split', 'wav_path'],
        header=None,
        nrows=100
    )

    splits['VoxCeleb1 ID'] = splits['wav_path'].apply(lambda p: p.split('/')[0])
    splits['wav_path'] = splits.apply(
        lambda r: get_wav_path(r['split'], r['wav_path']),
        axis='columns'
    )

    dataset = pd.merge(splits, meta, how='left', on='VoxCeleb1 ID', validate="m:1")

    dataset['spectrogram_path'] = dataset['wav_path'].apply(
        lambda p: os.path.join(NPY_PATH, convert_to_spectrogram_filename(p)))

    mel_config = configs['spectrogram']
    for _, row in dataset.iterrows():
        wav_path = row['wav_path']
        spectrogram_path = row['spectrogram_path']
        if row['split'] != TEST and not os.path.exists(spectrogram_path): # TODO: fix for testing
            mel_spectrogram = create_spectrogram(wav_path,
                                                 mel_config['offset_range'],
                                                 mel_config['sampling_rate'],
                                                 mel_config['sample_length'],
                                                 mel_config['fft_window'],
                                                 mel_config['hop_length'])

            persist_spectrogram(mel_spectrogram, spectrogram_path)

    return dataset


