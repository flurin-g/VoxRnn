from time import time
from os import path

import tensorflow as tf
import keras as ks
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_generator import DataGenerator, PreTrainDataGenerator
from vox_utils import get_all_sets, get_train_set
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
    klpq = ks.backend.sum(p * ks.backend.log(p / q), axis=-1)
    klqp = ks.backend.sum(q * ks.backend.log(q / p), axis=-1)
    return klpq + klqp


def kullback_leibler_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def kb_hinge_loss(y_true, y_pred):
    """
    y_true: binary label, 1 = same speaker
    y_pred: output of siamese net i.e. kullback-leibler distribution
    """
    # MARGIN = 1.
    # hinge = ks.backend.mean(ks.backend.softplus(MARGIN - y_pred), axis=-1)
    MARGIN = 3.
    hinge = ks.backend.mean(ks.backend.maximum(MARGIN - y_pred, 0.), axis=-1)
    return y_true * y_pred + (1 - y_true) * hinge


def kb_hinge_metric(y_true_targets, y_pred_KBL):
    THRESHOLD = 0.4
    THRESHOLD = ks.backend.cast(THRESHOLD, ks.backend.floatx())

    isMatch = ks.backend.less(y_pred_KBL, THRESHOLD)
    isMatch = ks.backend.cast(isMatch, ks.backend.floatx())

    isMatch = ks.backend.equal(y_true_targets, isMatch)
    isMatch = ks.backend.cast(isMatch, ks.backend.floatx())

    return ks.backend.mean(isMatch)


def create_lstm(units: int, gpu: bool, name: str, is_sequence: bool = True):
    if gpu:
        return ks.layers.CuDNNLSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS, name=name)
    else:
        return ks.layers.LSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS, unroll=True, name=name)


def build_model(mode: str = 'train') -> ks.Model:
    topology = TRAIN_CONF['topology']

    is_gpu = tf.test.is_gpu_available(cuda_only=True)

    input_layer = ks.layers.Input(shape=INPUT_DIMS)

    X = ks.layers.Bidirectional(create_lstm(topology['blstm1_units'],is_gpu,name='blstm_1'),input_shape=INPUT_DIMS)(input_layer)

    X = ks.layers.Bidirectional(create_lstm(topology['blstm2_units'], is_gpu, is_sequence=False, name='blstm_2'))(X)

    if mode == 'extraction':
        model = ks.models.Model(inputs=[input_layer], outputs=X)
        return model

    num_units = topology['dense1_units']
    X = ks.layers.Dense(num_units, activation='softplus', name='dense_1')(X)

    X = ks.layers.Dropout(topology['dropout2'])(X)

    num_units = topology['dense2_units']
    X = ks.layers.Dense(num_units, activation='softplus', name='dense_2')(X)

    num_units = topology['dense3_units']
    X = ks.layers.Dense(num_units, activation='softplus', name='dense_3')(X)

    model = ks.models.Model(inputs=[input_layer], outputs=X)
    return model


def build_siam():

    input_a = ks.Input(shape=INPUT_DIMS, name='input_a')
    input_b = ks.Input(shape=INPUT_DIMS, name='input_b')

    base_network = build_model()

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = ks.layers.Lambda(kullback_leibler_divergence,
                                output_shape=kullback_leibler_shape,
                                name='distance1')([processed_a, processed_b])

    model = ks.Model(inputs=[input_a,
                             input_b],
                     outputs=distance)
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
            histogram_freq=0,
            write_grads=True,
            write_images=True,
            write_graph=True
        )
    ]

    input_data = TRAIN_CONF['input_data']
    batch_size = input_data['batch_size']

    train_set, dev_set, test_set = get_all_sets(create_spectrograms)

    training_generator = DataGenerator(train_set, INPUT_DIMS, batch_size)

    val_data = training_generator.generate_batch(dev_set, batch_size, INPUT_DIMS[0], INPUT_DIMS[1], np.random.RandomState(1))

    siamese_net = build_siam()
    siamese_net.summary()
    siamese_net.fit_generator(generator=training_generator,
                              epochs=input_data['epochs'],
                              validation_data=val_data,
                              use_multiprocessing=True,
                              callbacks=callbacks)

    siamese_net.save_weights(weights_path, overwrite=True)


def build_pre_train_model(num_speakers: int):
    base_network = build_model()

    input_layer = ks.Input(shape=INPUT_DIMS, name='input')

    processed = base_network(input_layer)

    softmax = ks.layers.Dense(units=num_speakers, activation='softmax', name='softmax')(processed)

    model = ks.Model(inputs=input_layer, outputs=softmax)

    adam = build_optimizer()

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.summary()

    return model


def pre_train_model(create_spectrograms: bool = False, weights_path: str = WEIGHTS_PATH):
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

    train_set = get_train_set(create_spectrograms)
    unique_labels = train_set.speaker_id.unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    num_speakers = len(label_encoder.classes_)

    train, test = train_test_split(train_set, test_size=0.2, random_state=0, stratify=train_set['speaker_id'])

    training_generator = PreTrainDataGenerator(train,
                                               INPUT_DIMS,
                                               batch_size,
                                               label_encoder,
                                               num_speakers)

    val_data = training_generator.generate_batch(df=test,
                                                 batch_size=test.shape[0],
                                                 dim_0=INPUT_DIMS[0],
                                                 dim_1=INPUT_DIMS[1])

    pre_train_net = build_pre_train_model(num_speakers)

    pre_train_net.fit_generator(generator=training_generator,
                                epochs=input_data['epochs'],
                                validation_data=val_data,
                                use_multiprocessing=True)

    pre_train_net.save_weights(weights_path, overwrite=True)


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
