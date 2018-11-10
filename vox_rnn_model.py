import multiprocessing
from typing import Any, Union

import tensorflow as tf
import keras as ks

from data_generator import DataGenerator
from vox_utils import get_datasets


def build_optimizer(configs: dict):
    optimizer = None
    p = configs['topology']['optimizer']
    if p['type'] == 'adam':
        optimizer = ks.optimizers.Adam(p['learning_rate'],
                                       p['beta_1'],
                                       p['beta_2'],
                                       float(p['epsilon']),
                                       p['decay'])
    return optimizer


def kullback_leibler_divergence(vects):
    x, y = vects
    x = ks.backend.clip(x, ks.backend.epsilon(), 1)
    y = ks.backend.clip(y, ks.backend.epsilon(), 1)
    return ks.backend.sum(x * ks.backend.log(x / y), axis=-1)


def kullback_leibler_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def kb_hinge_loss(y_true, y_pred):
    """
    y_true: binary label, 1 = same speaker
    y_pred: output of siamese net i.e. kullback-leibler distribution
    """
    hinge = ks.backend.maximum(1. - y_pred, 0.)
    return y_true * y_pred + (1 - y_true) * hinge


def create_lstm(units: int, gpu: bool):
    if gpu:
        return ks.layers.CuDNNLSTM(units, return_sequences=True)
    else:
        return ks.layers.LSTM(units, return_sequences=True)


def build_model(input_dim: list, configs: dict, output_layer: str = 'layer8') -> ks.Model:
    topology = configs['topology']

    is_gpu = tf.test.is_gpu_available(cuda_only=True)

    layers = dict()

    X_input = ks.Input(shape=input_dim)

    layer1 = ks.layers.Bidirectional(create_lstm(topology['blstm1_units'], is_gpu))(X_input)

    layer2 = ks.layers.Dropout(topology['dropout1'])(layer1)

    layers['layer3'] = layer3 = ks.layers.Bidirectional(create_lstm(topology['blstm2_units'], is_gpu))(layer2)

    num_units = topology['dense1_units']
    layers['layer4'] = layer4 = ks.layers.Dense(num_units)(layer3)

    layer5 = ks.layers.Dropout(topology['dropout2'])(layer4)

    num_units = topology['dense2_units']
    layers['layer6'] = layer6 = ks.layers.Dense(num_units)(layer5)

    num_units = topology['dense3_units']
    layers['layer7'] = layer7 = ks.layers.Dense(num_units)(layer6)

    layers['layer8'] = layer8 = ks.layers.Dense(num_units, activation='softmax')(layer6)

    return ks.Model(inputs=X_input, outputs=layers[output_layer])


def build_siam(configs):
    input_data = configs['input_data']

    input_dim = [input_data['mel_spectrogram_x'],
                 input_data['mel_spectrogram_y']]

    base_network = build_model(input_dim, configs)

    input_a = ks.Input(shape=input_dim)
    input_b = ks.Input(shape=input_dim)

    model_a = base_network(input_a)
    model_b = base_network(input_b)

    distance = ks.layers.Lambda(kullback_leibler_divergence,
                                output_shape=kullback_leibler_shape)([model_a, model_b])

    model = ks.models.Model([input_a, input_b], distance)
    adam = build_optimizer(configs)
    model.compile(loss=kb_hinge_loss, optimizer=adam, metrics=['accuracy'])
    return model


def train_model(configs: dict, weights_path: str):
    cpu_cores = multiprocessing.cpu_count()

    input_data = configs['input_data']
    dim = (input_data['mel_spectrogram_x'],
           input_data['mel_spectrogram_y'])

    dataset = get_datasets(configs['input_data']['channels'])

    training_generator = DataGenerator(dataset,
                                       dim,
                                       input_data['batch_size'],
                                       input_data['batch_shuffle'])

    validation_generator = DataGenerator(dataset,
                                         dim,
                                         input_data['batch_size'],
                                         input_data['batch_shuffle'])

    siamese_net = build_siam(configs)

    siamese_net.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=True,
                              workers=cpu_cores)

    siamese_net.save_weights(weights_path, overwrite=False)


def build_embedding_extractor_net(configs: dict, output_layer: str, path: str):
    pass
    # create model with output layer3/4/6/7 to extract embeddings
    input_data = configs['input_data']

    input_dim = [input_data['mel_spectrogram_x'],
                 input_data['mel_spectrogram_y']]

    input_a = ks.Input(shape=input_dim)

    model = build_model(input_a, configs, output_layer)

    # load model
    model.load_weights(path, by_name=True)
