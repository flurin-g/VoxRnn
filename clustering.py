from typing import Tuple

import numpy as np
import pandas as pd
# ToDo: add mattplotlib to requirements.txt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from vox_rnn_model import build_embedding_extractor_net
from vox_utils import create_spectrogram
from vox_utils import get_test_set
from definitions import GLOBAL_CONF, TRAIN_CONF

MEL_CONFIG = GLOBAL_CONF['spectrogram']
INPUT_DATA = TRAIN_CONF['input_data']
embedding_extractor = build_embedding_extractor_net()


def load_segments(num_speaker: int, segments_per_speaker: int) -> Tuple[np.ndarray, list]:
    # test_set: pd.DataFrame = get_test_set()
    test_set = pd.read_csv('test_set_df.csv')

    # selects num_speakers, and from each speaker segments_per_speaker
    segments: pd.DataFrame = (test_set[test_set['speaker_id']
                              .isin(test_set['speaker_id']
                              .unique()[:num_speaker])]
                              .groupby('speaker_id')
                              .apply(lambda df: df.sample(segments_per_speaker)))

    embeddings = list()
    speaker_ids = list()

    for segment in segments.sample(frac=1).iterrows():
        spectrogram = create_spectrogram(segment[1]['wav_path'],
                                         MEL_CONFIG['offset'],
                                         MEL_CONFIG['sampling_rate'],
                                         None,
                                         MEL_CONFIG['fft_window'],
                                         MEL_CONFIG['hop_length'])

        # make spectrogram of segment a multiple of utterance length
        spectrogram = (spectrogram[0:(spectrogram.shape[0] - spectrogram.shape[0]
                                      % INPUT_DATA['mel_spectrogram_x']), :].reshape(-1,
                                                                                     INPUT_DATA['mel_spectrogram_x'],
                                                                                     INPUT_DATA['mel_spectrogram_y']))

        embeddings.extend(embedding_extractor.predict_on_batch(spectrogram))
        speaker_ids.extend([segment[1]['speaker_id']] * spectrogram.shape[0])

    return np.array(embeddings), speaker_ids

def plot_results(X, y):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == '__main__':
    X, y = load_segments(num_speaker=3, segments_per_speaker=2)
    print(type(X))
    plot_results(X, y)
