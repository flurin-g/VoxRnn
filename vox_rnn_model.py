import multiprocessing
import keras as ks

from pairwise_kl_divergence import pairwise_kl_divergence
from data_generator import DataGenerator
from vox_utils import get_datasets


def build_optimizer(configs: dict):
    optimizer = None
    p = configs['optimizer']
    if p['type'] == 'adam':
        optimizer = ks.optimizers.Adam(p['learning_rate'],
                                       p['beta_1'],
                                       p['beta_2'],
                                       float(p['epsilon']),
                                       p['decay'])
    return optimizer


def build_model(configs: dict, output_layer: str = 'layer8') -> ks.Model:
    """
    A spectrogram is the input and each discrete time-step is fed to one unit

    uses the CuDNNLSTM-Layer for optimized performance on GPU's.
    """
    input_data = configs['input_data']
    topology = configs['topology']

    layers = dict()

    input_dim = [input_data['mel_spectrogram_x'],
                 input_data['mel_spectrogram_y']]

    X_input = ks.Input(shape=input_dim)

    layer1 = ks.layers.Bidirectional(ks.layers.CuDNNLSTM(units=topology['blstm1']['units'],
                                                         return_sequences=True,
                                                         input_shape=input_dim))(X_input)

    layer2 = ks.layers.Dropout(topology['dropout1'])(layer1)

    layers['layer3'] = layer3 = ks.layers.Bidirectional(ks.layers.CuDNNLSTM(topology['blstm2']['units']))(layer2)

    num_units = topology['dense1']['units']
    layers['layer4'] = layer4 = ks.layers.Dense(num_units)(layer3)

    layer5 = ks.layers.Dropout(topology['dropout2'])(layer4)

    num_units = topology['dense2']['units']
    layers['layer6'] = layer6 = ks.layers.Dense(num_units)(layer5)

    num_units = topology['dense3']['units']
    layers['layer7'] = layer7 = ks.layers.Dense(num_units)(layer6)

    num_units = topology['dense3']['units']
    layers['layer8'] = layer8 = ks.layers.Dense(num_units, activation='softmax')(layer6)

    model = ks.Model(inputs=X_input,
                     outputs=layers[output_layer],
                     name='ResNet20')

    model.compile(loss=pairwise_kl_divergence,
                  optimizer=build_optimizer(topology),
                  metrics=['accuracy'])
    return model


def train_model(configs: dict, weights_path: str):
    cpu_cores = multiprocessing.cpu_count()

    input_data = configs['input_data']
    batch_size = input_data['batch_size']
    classes = configs['topology']['dense3']['units']

    dim = (input_data['mel_spectrogram_x'],
           input_data['mel_spectrogram_y'])

    data_splits, id_to_label = get_datasets(configs['input_data']['channels'])

    training_generator = DataGenerator(data_splits['train'],
                                       id_to_label,
                                       batch_size,
                                       dim,
                                       classes,
                                       shuffle=input_data['batch_shuffle'])

    validation_generator = DataGenerator(data_splits['dev'],
                                         id_to_label,
                                         batch_size,
                                         dim,
                                         classes,
                                         shuffle=input_data['batch_shuffle'])

    model = build_model(configs)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=cpu_cores)

    model.save_weights(weights_path, overwrite=False)


def build_embedding_extractor_net(configs: dict, output_layer: int, path: str):
    pass
    # create model with output layer3/4/6/7 to extract embeddings
    model = build_model(configs, output_layer)

    # load model
    model.load_weights(path, by_name=True)
