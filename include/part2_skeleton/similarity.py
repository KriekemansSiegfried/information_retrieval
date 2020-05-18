from itertools import product
from math import floor
import torch


def get_similarity_matrix(indices_x, indices_y):
    assert (len(indices_x) == len(indices_y))
    sim = torch.zeros(size=(len(indices_x), len(indices_x)))
    for (i, index_x), (j, index_y) in product(enumerate(indices_x), enumerate(indices_y)):
        if floor(index_x // 5) == floor(index_y // 5):
            sim[i, j] = 1
    return sim.cpu()
