from typing import Tuple

import numpy as np
import pandas as pd
# ToDo: add mattplotlib to requirements.txt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import sklearn as sk
from matplotlib import pyplot as plt

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

        utterance_embedding = np.mean(embedding_extractor.predict_on_batch(spectrogram), axis=0)
        embeddings.append(utterance_embedding)
        speaker_ids.extend([segment[1]['speaker_id']])

    return np.array(embeddings), speaker_ids


def create_dendrogram(X, y, segments):
    linked = linkage(X, 'single')
    labelList = range(1, segments + 1)

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=labelList,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()


def cluster_embeddings(X, y, num_speakers):
    cluster = sk.cluster.AgglomerativeClustering(n_clusters=num_speakers, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)

    for i in range(len(y)):
        print(str(cluster.labels_[i]) + ' - ' + str(y[i]))

    tsne = sk.manifold.TSNE(n_components=2, random_state=0)

    X = tsne.fit_transform(X)

    points = (X[:, 0], X[:, 1])

    plt.scatter(points[0], points[1], c=cluster.labels_, cmap='rainbow')

    for label, x, y in zip(y, points[0], points[1]):
        plt.annotate(label[-4:], xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(points[0].min() + 0.00002, points[0].max() + 0.00002)
        plt.ylim(points[1].min() + 0.00002, points[1].max() + 0.00002)
    plt.show()

    # plt.scatter(points[0], points[1], c=cluster.labels_, cmap='rainbow')
    # plt.annotate(y, (points[0], points[1]), fontsize=12)
    # plt.show()


if __name__ == '__main__':
    n_speak = 3
    n_seg = 5
    (X, y) = load_segments(num_speaker=n_speak, segments_per_speaker=n_seg)
    # create_dendrogram(X, y, n_speak * n_seg)
    cluster_embeddings(X, y, num_speakers=n_speak)
    # print(type(X))
    # plot_results(X, y, n_speak*n_seg)
