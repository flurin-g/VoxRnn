import sys
import yaml
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import CuDNNLSTM, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from VoxRnn.pairwise_kl_divergence import pairwise_kl_divergence


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


def build_embedding_train_model(frequency: int, time: int) -> Model:
    """
    uses the CuDNNLSTM-Layer for optimized performance on GPU's.
    A spectrogram is the input of one unit: total input = units x spectrogram

    :param frequency: number of frequency bins of mel-spectrogram
    :param time: number of discrete time-intervals of mel-spectrogram
    :return: keras model for training

    """
    configs = load_config()
    print(configs)
    num_speakers = configs['input_data']['num_speakers']

    X_input = create_input_layer(frequency, time)

    X = Bidirectional(CuDNNLSTM(configs['blstm1']['units'],
                           return_sequences=True))(X_input)

    X = Dropout(configs['dropout1'])(X)

    X = Bidirectional(CuDNNLSTM(configs['blstm1']['units']))(X)

    num_units = num_speakers * configs['dense1']['scaling']
    X = Dense(num_units)(X)

    X = Dropout(configs['dropout2'])(X)

    num_units = num_speakers * configs['dense2']['scaling']
    X = Dense(num_units)(X)

    num_units = num_speakers
    X = Dense(num_units)(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet20')

    model.compile(loss=pairwise_kl_divergence,
                  optimizer=build_optimizer(configs),
                  metrics=['accuracy'])
    return model


def get_data() -> tuple:
    """
    returns tuple of numpy arrays containing respective data
    :return: train_x, train_y, validate_x, validate_y
    """
    pass


def train_model():
    train_x, train_y, validate_x, validate_y = get_data()


if __name__ == "__main__":
    model = build_embedding_train_model(40, 128)
    print(model)
