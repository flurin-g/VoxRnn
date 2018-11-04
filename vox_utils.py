from pandas import read_fwf
from numpy import ndarray

# ToDo: write methods that gets spectrogram from voxceleb, and compiles them into train and test sets
def load_labels():
    labels = read_fwf('../Data/iden_split.txt')
    print(labels)


def get_spectrograms(label: str) -> ndarray:
#                  - get corresponding label and add to datastructure
    pass

def build_x(labels: list) -> ndarray:
    x = list()
    # for each sample: - create spectrogram with random start-point and add to a data-structure
    for label in labels:
        x.append(get_spectrograms(label))


def get_datasets() -> tuple:
    """
    Generates train_x, train_y, validate_x, validate_y from a given dataset
    :return: tuple containing train and test sets as ndarray
    """

    labels = load_labels()

    train_y, dev_y, test_y = split_labels(labels)

    train_x = build_x(train_y)
    dev_x = build_x(dev_y)
    test_x = build_x(test_y)


    return train_x, train_y, dev_x, validate_y, test_x, test_y



if __name__ == "__main__":
    load_labels()