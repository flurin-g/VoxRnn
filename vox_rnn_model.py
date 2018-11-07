import multiprocessing
import sys
import yaml
import tensorflow.keras as ks

from definitions import TRAIN_CONF, WEIGHTS_PATH
from pairwise_kl_divergence import pairwise_kl_divergence
from data_generator import DataGenerator
from vox_utils import get_datasets

def create_input_layer(batch_size: int, time: int, frequency: int, channels: int) -> ks.Input:
    return ks.Input(shape=[batch_size, time, frequency, channels])


def build_optimizer(configs: dict):
    p = configs['optimizer']
    if p['type'] == 'adam':
        optimizer = Adam(p['learning_rate'],
                         p['beta_1'],
                         p['beta_2'],
                         p['epsilon'],
                         p['decay'])
    return optimizer


def build_model(configs: dict, num_speakers: int, output_layer: str = 'layer7') -> ks.Model:
    """
    A spectrogram is the input and each discrete time-step is fed to one unit

    uses the CuDNNLSTM-Layer for optimized performance on GPU's.
    """
    input_data = configs['input_data']
    topology = configs['topology']

    layers = dict()

    input_dim = [input_data['batch_size'],
                 input_data['mel_spectrogram_x'],
                 input_data['mel_spectrogram_y'],
                 input_data['channels']]

    X_input = create_input_layer(*input_dim)

    layer1 = ks.layers.Bidirectional(ks.layers.CuDNNLSTM(topology['blstm1']['units'],
                                     return_sequences=True,
                                     input_shape=input_dim))(X_input)

    layer2 = ks.layers.Dropout(topology['dropout1'])(layer1)

    layers['layer3'] = layer3 = ks.wrappersBidirectional(ks.layers.CuDNNLSTM(topology['blstm1']['units']))(layer2)

    num_units = num_speakers * topology['dense1']['scaling']
    layers['layer4'] = layer4 = ks.layers.Dense(num_units)(layer3)

    layer5 = ks.layers.Dropout(topology['dropout2'])(layer4)

    num_units = num_speakers * topology['dense2']['scaling']
    layers['layer6'] = layer6 = ks.layers.Dense(num_units)(layer5)

    num_units = num_speakers
    layers['layer7'] = layer7 = ks.layers.Dense(num_units)(layer6)

    model = ks.Model(inputs=X_input, outputs=output_layer, name='ResNet20')

    model.compile(loss=pairwise_kl_divergence,
                  optimizer=build_optimizer(topology),
                  metrics=['accuracy'])
    return model


def train_model(configs: dict, weights_path: str):
    cpu_cores = multiprocessing.cpu_count()

    input_data = configs['input_data']

    dim = (input_data['mel_spectrogram_x'],
           input_data['mel_spectrogram_y'],
           input_data['channels'])

    data_splits, id_to_label, num_speakers = get_datasets(configs['input_data']['channels'])

    training_generator = DataGenerator(data_splits['train'],
                                       id_to_label,
                                       input_data['batch_size'],
                                       dim=dim,
                                       n_classes=num_speakers,
                                       shuffle=input_data['batch_shuffle'])

    validation_generator = DataGenerator(data_splits['dev'],
                                         id_to_label,
                                         input_data['batch_size'],
                                         dim=dim,
                                         n_classes=num_speakers,
                                         shuffle=input_data['batch_shuffle'])

    model = build_model(configs)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=cpu_cores)

    model.save_weights(weights_path, overwrite=False)


def build_embedding_extractor_net(configs: dict, num_speakers: int, output_layer: int, path: str):
    pass
    # create model with output layer3/4/6/7 to extract embeddings
    model = build_model(configs, num_speakers, output_layer)

    # load model
    model.load_weights(path, by_name=True)


if __name__ == "__main__":
    configs = TRAIN_CONF
    train_model(configs, WEIGHTS_PATH)
