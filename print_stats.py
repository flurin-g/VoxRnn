import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import keras as ks


import vox_rnn_model

from definitions import GLOBAL_CONF
SPECS = GLOBAL_CONF["spectrogram"]


def plot_spectrogram():
    S = np.load("/Users/flurin/repos/hs18/pa/VoxRnn/Data/VoxCeleb1/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav.npy")
    S = np.rot90(S, k=3)
    print(S.shape)
    plt.figure(figsize=(12, 8))

    librosa.display.specshow(data=S,
                             sr=SPECS["sampling_rate"],
                             hop_length=SPECS["hop_length"],
                             y_axis='mel',
                             fmax=8000,
                             x_axis='time')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


def plot_model():
    model = vox_rnn_model.build_model(1251)
    ks.utils.plot_model(model=model, to_file='base_network.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    #plot_spectrogram()
    plot_model()
