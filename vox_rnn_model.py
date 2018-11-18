import tensorflow as tf
import keras as ks

from data_generator import DataGenerator
from vox_utils import get_all_sets
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
    hinge = ks.backend.mean(ks.backend.square(ks.backend.maximum(MARGIN - y_true * y_pred, 0.)), axis=-1)
    return y_true * y_pred + (1 - y_true) * hinge


def create_lstm(units: int, gpu: bool, is_sequence: bool = True):
    if gpu:
        return ks.layers.CuDNNLSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS)
    else:
        return ks.layers.LSTM(units, return_sequences=is_sequence, input_shape=INPUT_DIMS)


def build_model(mode: str = 'train_pairs') -> ks.Model:
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

    if mode == 'embedding_extraction':
        return model

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

    distance = ks.layers.Lambda(kullback_leibler_divergence,
                                output_shape=kullback_leibler_shape)([processed_a, processed_b])

    model = ks.Model(inputs=[input_a, input_b], outputs=distance)
    adam = build_optimizer()
    model.compile(loss=kb_hinge_loss, optimizer=adam, metrics=['accuracy'])
    return model


def train_model(create_spectrograms: bool = False, weights_path: str = WEIGHTS_PATH):
    input_data = TRAIN_CONF['input_data']
    train_set, dev_set, test_set = get_all_sets(create_spectrograms)

    training_generator = DataGenerator(train_set,
                                       INPUT_DIMS,
                                       input_data['batch_size'],
                                       input_data['batch_shuffle'])

    validation_generator = DataGenerator(dev_set,
                                         INPUT_DIMS,
                                         input_data['batch_size'],
                                         input_data['batch_shuffle'])

    siamese_net = build_siam()

    siamese_net.summary()
    # TODO: set epochs and implement tensorboard
    siamese_net.fit_generator(generator=training_generator,
                              epochs=input_data['epochs'],
                              validation_data=validation_generator)

    siamese_net.save_weights(weights_path, overwrite=True)


def build_embedding_extractor_net():
    base_network = build_model('embedding_extraction')

    input_layer = ks.Input(shape=INPUT_DIMS)

    processed = base_network(input_layer)

    adam = build_optimizer()

    processed.compile(loss=kb_hinge_loss, optimizer=adam, metrics=['accuracy'])

    processed.load_weights(WEIGHTS_PATH, by_name=True)

    return processed
