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


def euclidean_distance(vects):
    x, y = vects
    return ks.backend.sqrt(ks.backend.sum(ks.backend.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
    margin = 1
    return ks.backend.mean(y_true * ks.backend.square(y_pred) + (1 - y_true) * ks.backend.square(ks.backend.maximum(margin - y_pred, 0)))


def create_lstm(units: int, gpu: bool, is_sequence: bool = True):
    if gpu:
        return ks.layers.CuDNNLSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS)
    else:
        return ks.layers.LSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS)


def build_model(output_layer: str = 'layer8') -> ks.Model:
    topology = TRAIN_CONF['topology']

    is_gpu = tf.test.is_gpu_available(cuda_only=True)

    model = ks.Sequential()

    model.add(ks.layers.Bidirectional(create_lstm(topology['blstm1_units'], is_gpu),
                                      input_shape=INPUT_DIMS))

    model.add(ks.layers.Dropout(topology['dropout1']))

    model.add(ks.layers.Bidirectional(create_lstm(topology['blstm2_units'],
                                                  is_gpu,
                                                  is_sequence=False)))

    num_units = topology['dense1_units']
    model.add(ks.layers.Dense(num_units))

    model.add(ks.layers.Dropout(topology['dropout2']))

    num_units = topology['dense2_units']
    model.add(ks.layers.Dense(num_units))

    num_units = topology['dense3_units']
    model.add(ks.layers.Dense(num_units))

    model.add(ks.layers.Dense(num_units, activation='softmax'))

    return model


# ks.Model(inputs=X_input, outputs=layers[output_layer])
def build_siam():
    base_network = build_model()

    input_a = ks.Input(shape=INPUT_DIMS)
    input_b = ks.Input(shape=INPUT_DIMS)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = ks.layers.Lambda(euclidean_distance,
                                output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = ks.Model(inputs=[input_a, input_b], outputs=distance)
    adam = build_optimizer()
    model.compile(loss=contrastive_loss, optimizer=adam, metrics=['accuracy'])
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

    siamese_net.summary()
    # TODO: set epochs and implement tensorboard
    siamese_net.fit_generator(generator=training_generator,
                              steps_per_epoch=input_data['epochs'],
                              validation_data=validation_generator)

    siamese_net.save_weights(weights_path, overwrite=True)


def build_embedding_extractor_net(output_layer: str):
    pass
    # create model with output layer3/4/6/7 to extract embeddings
    model = build_model(output_layer)

    # load model
    model.load_weights(WEIGHTS_PATH, by_name=True)
