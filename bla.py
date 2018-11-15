import numpy as np
import os

from definitions import NPY_PATH


def file_name_temp(file_name_raw: str):
    return os.path.join(NPY_PATH, file_name_raw.replace('/', '-').replace('.', '-'))


if __name__ == '__main__':
    hey = np.load(file_name_temp('id10001/utrA-v8pPm4/00001.wav'))

    print(hey)
