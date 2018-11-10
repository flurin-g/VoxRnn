import itertools
import numpy as np

hey = np.arange(100)
ho = list(itertools.combinations(hey, 2))

print(ho)