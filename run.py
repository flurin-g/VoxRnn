import argparse

from vox_rnn_model import train_model, pre_train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a speaker embedding model.')

    parser.add_argument('-S', '--create-spectrograms', dest='create_spectrograms', default=False,
                        action='store_true')

    parser.add_argument('-M', '--mode', type=str, dest='training_mode', default="train")

    args = parser.parse_args()

    if args.training_mode == "pre-train":
        pre_train_model(create_spectrograms=args.create_spectrograms)
    else:
        train_model(create_spectrograms=args.create_spectrograms)
