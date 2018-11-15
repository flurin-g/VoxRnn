import multiprocessing
from typing import Any, Union

import tensorflow as tf
import keras as ks

from data_generator import DataGenerator
from vox_utils import get_dataset
from definitions import TRAIN_CONF, WEIGHTS_PATH

INPUT_DIMS = [TRAIN_CONF['input_data']['mel_spectrogram_x'],
              TRAIN_CONF['input_data']['mel_spectrogram_y']]


def build_optimizer():
    optimizer = None
    p = TRAIN_CONF['topology']['optimizer']
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
    MARGIN = 3.
    hinge = ks.backend.maximum(MARGIN - y_pred, 0.)
    return y_true * y_pred + (1 - y_true) * hinge


def create_lstm(units: int, gpu: bool, is_sequence: bool=True):
    if gpu:
        return ks.layers.CuDNNLSTM(units, return_sequences=is_sequence)
    else:
        return ks.layers.LSTM(units, return_sequences=is_sequence)


def build_model(output_layer: str = 'layer8') -> ks.Model:
    topology = TRAIN_CONF['topology']

    is_gpu = tf.test.is_gpu_available(cuda_only=True)

    layers = dict()

    X_input = ks.Input(shape=INPUT_DIMS)

    layer1 = ks.layers.Bidirectional(create_lstm(topology['blstm1_units'], is_gpu))(X_input)

    layer2 = ks.layers.Dropout(topology['dropout1'])(layer1)

    layers['layer3'] = layer3 = ks.layers.Bidirectional(create_lstm(topology['blstm2_units'],
                                                                    is_gpu,
                                                                    is_sequence=False))(layer2)

    num_units = topology['dense1_units']
    layers['layer4'] = layer4 = ks.layers.Dense(num_units)(layer3)

    layer5 = ks.layers.Dropout(topology['dropout2'])(layer4)

    num_units = topology['dense2_units']
    layers['layer6'] = layer6 = ks.layers.Dense(num_units)(layer5)

    num_units = topology['dense3_units']
    layers['layer7'] = layer7 = ks.layers.Dense(num_units)(layer6)

    layers['layer8'] = layer8 = ks.layers.Dense(num_units, activation='softmax')(layer7)

    return ks.Model(inputs=X_input, outputs=layers[output_layer])


def build_siam():
    base_network = build_model()

    input_a = ks.Input(shape=INPUT_DIMS)
    input_b = ks.Input(shape=INPUT_DIMS)

    model_a = base_network(input_a)
    model_b = base_network(input_b)

    distance = ks.layers.Lambda(kullback_leibler_divergence,
                                output_shape=kullback_leibler_shape)([model_a, model_b])

    model = ks.models.Model([input_a, input_b], distance)
    adam = build_optimizer()
    model.compile(loss=kb_hinge_loss, optimizer=adam, metrics=['accuracy'])
    return model


def train_model(weights_path: str = WEIGHTS_PATH):
    cpu_cores = multiprocessing.cpu_count()

    input_data = TRAIN_CONF['input_data']

    dataset = None

    training_generator = DataGenerator(dataset,
                                       INPUT_DIMS,
                                       input_data['batch_size'],
                                       input_data['batch_shuffle'])

    validation_generator = DataGenerator(dataset,
                                         INPUT_DIMS,
                                         input_data['batch_size'],
                                         input_data['batch_shuffle'])

    siamese_net = build_siam()
    # TODO: set epochs and implement tensorboard
    siamese_net.fit_generator(generator=training_generator,
                              validation_data=validation_generator)

    siamese_net.save_weights(weights_path, overwrite=True)


def build_embedding_extractor_net(output_layer: str):
    pass
    # create model with output layer3/4/6/7 to extract embeddings
    model = build_model(output_layer)

    # load model
    model.load_weights(WEIGHTS_PATH, by_name=True)
