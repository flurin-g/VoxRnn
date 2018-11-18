import argparse

from vox_rnn_model import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a speaker embedding model.')

    parser.add_argument('-S', '--create-spectrograms', dest='create_spectrograms', default=False,
                        action='store_true')

    args = parser.parse_args()

    train_model(
        create_spectrograms=args.create_spectrograms)
