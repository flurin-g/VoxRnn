import os

from itertools import combinations
import librosa as lr
import numpy as np
import pandas as pd

from definitions import GLOBAL_CONF, NPY_PATH, VOX_DEV_WAV, VOX_TEST_WAV, SPLITS, VOX_PAIRS

TRAIN = 1  # in orig file + 1, but because lists are zero based...
DEV = 2
TEST = 3
NUM_DATA_SETS = 2
SAME_SPEAKER = 1
DIFFERENT_SPEAKER = 0
TRAIN_SET_SIZE = 950000
DEV_SET_SIZE = 50000
NUM_PER_POI = 100


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


def create_pairs():
    path = get_path(SPLITS)
    speaker_ids = pd.read_csv(path,
                              sep="\s|/",
                              names=['data_set', 'id', 'folder', 'filename'],
                              usecols=['data_set', 'id'],
                              header=None,
                              engine="python")

    utterance_path = pd.read_csv(path, sep=' ',
                                 names=['data_set', 'path'],
                                 usecols=['path'],
                                 header=None)

    veri_split = pd.concat([speaker_ids, utterance_path], axis=1)

    train_split = veri_split[veri_split['data_set'] == TRAIN]
    dev_split = veri_split[veri_split['data_set'] == DEV]

    write_same_pairs_file(train_split,
                          TRAIN_SET_SIZE,
                          NUM_PER_POI,
                          'same_speakers_train.csv')

    write_same_pairs_file(dev_split,
                          DEV_SET_SIZE,
                          NUM_PER_POI,
                          'same_speakers_dev.csv')


def write_same_pairs_file(split_frame, train_set_size, num_per_poi, csv_name):
    grouped = split_frame.groupby('id')
    pairs_list = list()
    i = 0
    for name, group in grouped:
        j = 0
        for index in list(combinations(group.index, 2)):
            pairs_list.append([SAME_SPEAKER, group.loc[index[0], 'path'], group.loc[index[1], 'path']])

            i += 1
            j += 1
            if j >= num_per_poi:
                break
        if i >= train_set_size / 2:
            break
    same_speaker_df = pd.DataFrame(data=pairs_list, columns=['y_label', 'utterance_a', 'utterance_b'])
    same_speaker_df.to_csv(csv_name)
