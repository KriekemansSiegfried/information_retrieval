import numpy as np
from scipy.sparse import csr_matrix, save_npz

from include.bow import one_hot


def output_captions(captions, tokens, file_name="include/data/caption_features", verbose=True, n_rows=None):
    """
    :param captions:
    :param tokens:
    :param file_name:
    :param verbose:
    :param n_rows:
    :return:
    """

    if n_rows is None:
        n_rows = len(captions)
    # store arrays
    result_arr = []
    for i in range(n_rows):
        out = one_hot.convert_to_bow(captions[i], tokens)
        # save as a sparse vector (note this is not the same as the sparse_matrix)
        captions[i].features = csr_matrix(out)
        result_arr.append(out)
        if i % 10000 == 0 and verbose:
            print(i)
    result_arr = np.vstack(result_arr)
    sparse_matrix = csr_matrix(result_arr)
    save_npz(file_name, sparse_matrix)
