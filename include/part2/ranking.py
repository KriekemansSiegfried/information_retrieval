import numpy as np
from scipy.spatial.distance import hamming


def find_nearest(input, data_set, nr=10):
    input = input.round()
    data = data_set[1].round()
    distances = np.zeros(len(data), dtype=np.int)
    for i in range(len(data)):
        distances[i] = hamming(input, data[i, :])
    indices = np.argsort(distances, kind='quicksort')
    result_list = [data_set[0][i] for i in np.flip(indices)]
    return result_list[0:nr]
