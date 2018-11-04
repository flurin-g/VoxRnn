import sys
import yaml
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import CuDNNLSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from pairwise_kl_divergence import pairwise_kl_divergence
from vox_utils import get_datasets


def create_input_layer(frequency: int, time: int) -> Input:
    return Input(shape=[frequency, time])


def load_config():
    with open("train_config.yaml", 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print("error parsing file")
            sys.exit(1)


def build_optimizer(configs: dict):
    p = configs['optimizer']
    if p['type'] == 'adam':
        optimizer = Adam(p['learning_rate'],
                         p['beta_1'],
                         p['beta_2'],
                         p['epsilon'],
                         p['decay'])
    return optimizer


def build_embedding_train_model(frequency: int, units: int) -> Model:
    """
    Builds the model used to train on data, result: weights which can be loaded
    in a modified model to extract embeddings from a given utterance

    A spectrogram is the input and each discrete time-step is fed to one unit

    uses the CuDNNLSTM-Layer for optimized performance on GPU's.

    :param frequency: number of frequency bins of mel-spectrogram
    :param units: each unit is one discrete time step in the spectrogram, 10 ms = 1 step
    :return: keras model for training
    """
    configs = load_config()
    print(configs)
    num_speakers = configs['input_data']['num_speakers']

    X_input = create_input_layer(frequency, units)

    layer1 = Bidirectional(CuDNNLSTM(configs['blstm1']['units'],
                           return_sequences=True))(X_input)

    layer2 = Dropout(configs['dropout1'])(layer1)

    layer3 = Bidirectional(CuDNNLSTM(configs['blstm1']['units']))(layer2)

    num_units = num_speakers * configs['dense1']['scaling']
    layer4 = Dense(num_units)(layer3)

    layer5 = Dropout(configs['dropout2'])(layer4)

    num_units = num_speakers * configs['dense2']['scaling']
    layer6 = Dense(num_units)(layer5)

    num_units = num_speakers
    layer7 = Dense(num_units)(layer6)

    model = Model(inputs=X_input, outputs=layer7, name='ResNet20')

    model.compile(loss=pairwise_kl_divergence,
                  optimizer=build_optimizer(configs),
                  metrics=['accuracy'])
    return model


def train_model(model: Model):
    # ToDo: implement stanford streaming
    pass


if __name__ == "__main__":
    model = build_embedding_train_model(40, 128)
    train_model(model)
