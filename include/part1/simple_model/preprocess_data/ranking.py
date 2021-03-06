import numpy as np
import pandas as pd
from include.util.util import print_progress_bar
from scipy import spatial


def rank_images(true_label,
                predictions,
                scoring_function='mse',
                k=10,
                verbose=True,
                batch_sizes_equal=True):
    """

    :param true_label:
    :param predictions:
    :param scoring_function:
    :param k:
    :param verbose:
    :param batch_sizes_equal:
    :return:
    """

    ranking = {}
    n = len(predictions)
    for i in range(n):
        if verbose:
            print_progress_bar(i=i, maximum=n)
        scores_ = []
        for j in range(len(true_label)):
            if scoring_function == 'cosine':
                score = spatial.distance.cosine(predictions[i, :], true_label[j, 1:].astype(float))
            elif scoring_function == 'mse':
                # element wise mse
                score = ((predictions[i, :] - true_label[j, 1:].astype(float)) ** 2).mean(axis=None)
            else:
                print("metric not available, available metrics include mse and cosine")
            scores_.append((true_label[j, 0], score))
        # save lowest k id's and scores in ascending (score) order
        if batch_sizes_equal:
            ranking[true_label[i, 0]] = (sorted(scores_, key=lambda x: x[1]))[0:k]
        else:
            ranking["search_term_" + str(i)] = (sorted(scores_, key=lambda x: x[1]))[0:k]
    return ranking


def compute_average_precision(dic, verbose=False):

    """

    :param dic:
    :param verbose:
    :return:
    """

    store_idx = {}
    counter = 0
    n = len(dic.items())
    for key, value in dic.items():
        if verbose:
            print_progress_bar(i=counter, maximum=n)
        list_keys = [item[0] for item in value]
        # check if ground true label (image_id) is in in the first k (=10) predicted labels (image_id)
        if key in list_keys:
            np_array = np.array(list_keys)
            # get indice where ground true label (image_id) == predicted label (image_id)
            item_index = np.where(np_array == key)[0][0]
            # compute the average precision
            store_idx[key] = 1 / (item_index + 1)
        else:
            # ground true label (image_id) is not in in the first k (=10) predicted labels (image_id)
            # ==> average precision = 0
            store_idx[key] = 0
        counter += 1
    return pd.DataFrame.from_dict(store_idx, orient='index', columns=['score'])






