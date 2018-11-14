import os

import librosa as lr
import numpy as np
import pandas as pd

from definitions import GLOBAL_CONF, NPY_PATH, VOX_DEV_WAV, VOX_TEST_WAV, SPLITS

TRAIN = 1  # in orig file + 1, but because lists are zero based...
DEV = 2
TEST = 3
NUM_DATA_SETS = 2


def get_path(name: str) -> str:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    name = os.path.join(current_directory, name)
    return name


def convert_to_filename(path_name: str) -> str:
    return path_name.replace('.wav', '').replace('/', '-').replace('.', '-')


def write_to_disk(mel_spectrogram: np.ndarray, dest_dir: str, file_id: str):
    path = os.path.join(dest_dir, file_id)
    np.save(path, mel_spectrogram, allow_pickle=False)


def load_rows(name: str) -> pd.DataFrame:
    path = get_path(name)
    return pd.read_csv(path, sep=' ', names=['data_set', 'path'], header=None)


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


def create_all_spectrograms():
    rows = load_rows(SPLITS)

    for _, row in rows.iterrows():
        if row['data_set'] == TRAIN or row['data_set'] == DEV:
            wav_path = os.path.join(VOX_DEV_WAV, row['path'])
        else:
            wav_path = os.path.join(VOX_TEST_WAV, row['path'])

        mel_spectrogram = create_spectrogram(wav_path, **GLOBAL_CONF['spectrogram'])

        file_id: str = convert_to_filename(row['path'])

        write_to_disk(mel_spectrogram, NPY_PATH, file_id)
