from time import time
from os import path

import tensorflow as tf
import keras as ks
import numpy as np

from data_generator import DataGenerator
from vox_utils import get_all_sets
from definitions import TRAIN_CONF, WEIGHTS_PATH, LOG_DIR

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

    if p['type'] == 'rms':
        optimizer = ks.optimizers.RMSprop()

    return optimizer


def kullback_leibler_divergence(speakers):
    p, q = speakers
    p = p + ks.backend.epsilon()
    q = q + ks.backend.epsilon()
    return ks.backend.sum(p * ks.backend.log(p / q), axis=-1)


def kullback_leibler_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def kb_hinge_loss(y_true, y_pred):
    """
    y_true: binary label, 1 = same speaker
    y_pred: output of siamese net i.e. kullback-leibler distribution
    """
    MARGIN = 1.
    hinge = ks.backend.mean(ks.backend.softplus(MARGIN - y_pred), axis=-1)
    return y_true * y_pred + (1 - y_true) * hinge


def kb_hinge_metric(y_true_targets, y_pred_KBL):
    THRESHOLD = 0.4
    
    isMatch = ks.backend.less(y_pred_KBL, THRESHOLD)
    isMatch = ks.backend.cast(isMatch, ks.backend.floatx())

    isMatch = ks.backend.equal(y_true_targets, isMatch)
    isMatch = ks.backend.cast(isMatch, ks.backend.floatx())

    return ks.backend.mean(isMatch)


def create_lstm(units: int, gpu: bool, name: str, is_sequence: bool = True):
    if gpu:
        return ks.layers.CuDNNLSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS, name=name)
    else:
        return ks.layers.LSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS, name=name)


def build_model(mode: str = 'train') -> ks.Model:
    topology = TRAIN_CONF['topology']

    is_gpu = tf.test.is_gpu_available(cuda_only=True)

    model = ks.Sequential(name='base_network')

    model.add(
        ks.layers.Bidirectional(create_lstm(topology['blstm1_units'], is_gpu, name='blstm_1'), input_shape=INPUT_DIMS))

    model.add(ks.layers.Dropout(topology['dropout1']))

    model.add(ks.layers.Bidirectional(create_lstm(topology['blstm2_units'], is_gpu, is_sequence=False, name='blstm_2')))

    if mode == 'extraction':
        return model

    num_units = topology['dense1_units']
    model.add(ks.layers.Dense(num_units, activation='softplus', name='dense_1'))

    model.add(ks.layers.Dropout(topology['dropout2']))

    num_units = topology['dense2_units']
    model.add(ks.layers.Dense(num_units, activation='softplus', name='dense_2'))

    num_units = topology['dense3_units']
    model.add(ks.layers.Dense(num_units, activation='softplus', name='dense_3'))

    model.add(ks.layers.Dense(num_units, activation='softplus', name='output'))

    #model.add(ks.layers.Softmax(num_units, name='softmax'))

    return model


def build_siam():
    base_network = build_model()

    input_a = ks.Input(shape=INPUT_DIMS, name='input_a')
    input_b = ks.Input(shape=INPUT_DIMS, name='input_b')

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance1 = ks.layers.Lambda(kullback_leibler_divergence,
                                 output_shape=kullback_leibler_shape,
                                 name='distance1')([processed_a, processed_b])

    distance2 = ks.layers.Lambda(kullback_leibler_divergence,
                                 output_shape=kullback_leibler_shape,
                                 name='distance2')([processed_b, processed_a])

    distance = ks.layers.Add(name='distance_add')([distance1, distance2])

    model = ks.Model(inputs=[input_a, input_b], outputs=distance)
    adam = build_optimizer()
    model.compile(loss=kb_hinge_loss,
                  optimizer=adam,
                  metrics=['accuracy', kb_hinge_metric])
    return model


def train_model(create_spectrograms: bool = False, weights_path: str = WEIGHTS_PATH):
    model_dir = path.dirname(WEIGHTS_PATH)
    checkpoint_pattern = path.join(model_dir, 'weights.{epoch:02d}-{val_loss:.2f}-' + str(time()) + '.hdf5')

    callbacks = [
        ks.callbacks.ProgbarLogger('steps'),
        ks.callbacks.ModelCheckpoint(checkpoint_pattern),
        ks.callbacks.TensorBoard(
            LOG_DIR,
            histogram_freq=1,
            write_grads=True,
            write_images=True,
            write_graph=True
        )
    ]

    input_data = TRAIN_CONF['input_data']
    batch_size = input_data['batch_size']

    train_set, dev_set, test_set = get_all_sets(create_spectrograms)

    training_generator = DataGenerator(train_set, INPUT_DIMS, batch_size)

    val_data = DataGenerator.generate_batch(dev_set, batch_size, INPUT_DIMS[0], INPUT_DIMS[1], np.random.RandomState(1))

    siamese_net = build_siam()
    siamese_net.summary()
    siamese_net.fit_generator(generator=training_generator,
                              epochs=input_data['epochs'],
                              validation_data=val_data,
                              use_multiprocessing=True,
                              callbacks=callbacks,
                              workers=4)

    siamese_net.save_weights(weights_path, overwrite=True)


def build_embedding_extractor_net():
    # ks.layers.core.K.set_learning_phase(0)

    base_network = build_model('extraction')

    input_layer = ks.Input(shape=INPUT_DIMS, name='input')

    processed = base_network(input_layer)

    model = ks.Model(input=input_layer, output=processed)

    optimizer = build_optimizer()

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    model.load_weights(WEIGHTS_PATH, by_name=True)

    return model
