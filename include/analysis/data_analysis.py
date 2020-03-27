import numpy as np
import pandas as pd
from include.analysis.visualize import print_progress_bar
from scipy import spatial


def rank_images(true_label, predictions, scoring_function='mse', k=10, verbose=True,
                batch_sizes_equal=True):
    """

    :param true_label:
    :param predictions:
    :param scoring_function:
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
    #print("\n")
    return ranking


def comput_average_precision(dic, verbose=False):
    """
    :param dic:
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


def rank_efficient(caption_features, image_features, id_included=False, k=10):
    id_list = None

    if id_included:
        id_list = image_features[:, 0]
        image_features = image_features[:, 1:]

    caption_count = caption_features.shape[0]
    image_count = image_features.shape[0]

    if k is None:
        k = image_count

    caption_norms = []
    image_norms = []
    for index in range(0, caption_count):
        r = caption_features[index, :]
        caption_norms.append(np.norm(r.astype(np.float)))
        # caption_norms.append(norm(caption_features[index, :]))
    for index in range(0, image_count):
        r = image_features[index, :]
        image_norms.append(np.norm(r.astype(np.float)))
        # image_norms.append(norm(image_features[index, :]))

    base = [range(0, image_count)]
    empty = [None] * image_count
    result = np.array([[0]*k]*caption_count)
    cos_sim = np.c_[np.transpose(base), np.transpose([empty])]
    for i in range(0, caption_count):
        caption_norm = caption_norms[i]
        for j in range(0, image_count):
            a = caption_features[i, :]
            b = image_features[j, :]
            cos_sim[j, 1] = - np.dot(a.astype(np.float), b.astype(np.float)) / (caption_norm * image_norms[j])

            # cos_sim[j, 1] = - np.dot(caption_features[i, :], image_features[j, :])/(caption_norm*image_norms[j])
        ranked = cos_sim[cos_sim[:, 1].argsort(kind='quicksort')][:k, 0]
        result[i] = ranked

    if not id_included:
        return result

    list_result = []
    for i in range(0, result.shape[0]):
        new_list = []
        for j in range(0, k):
            new_list.append(id_list[result[i, j]])
        list_result.append(np.array(new_list))
    return np.array(list_result)


def convert_to_dataset(pairs):
    """ convert all pairs to dataset with labels 0/1: negative/positive"""

    labels = []
    caption_features_set = []
    image_features_set = []

    for (key, positive, negative) in pairs:
        image_features = key.features
        positive_caption_features = positive.features
        negative_caption_features = negative.features

        # add positive example
        image_features_set.append(image_features)
        caption_features_set.append(positive_caption_features)
        labels.append(np.float64(1))

        # add negative example
        image_features_set.append(image_features)
        caption_features_set.append(negative_caption_features)
        labels.append(np.float64(0))

    image_features_set = np.stack(image_features_set, axis=0)
    caption_features_set = np.stack(caption_features_set, axis=0)

    print('caption_feature_set -> {}'.format(caption_features_set.shape))

    labels = np.stack(labels, axis=0)

    return [image_features_set, caption_features_set], labels


def convert_to_triplet_dataset(triplets):

    labels = []
    caption_features_set_neg = []
    caption_features_set_pos = []
    image_features_set = []

    for (key, positive, negative) in triplets:
        image_features_set.append(key.features)
        caption_features_set_pos.append(positive.features)
        caption_features_set_neg.append(negative.features)
        labels.append(0)  # dummy label

        # yield [negative.features, positive.features, key.features], np.float64(1)

    image_features_set = np.stack(image_features_set, axis=0)
    caption_features_set_pos = np.stack(caption_features_set_pos, axis=0)
    caption_features_set_neg = np.stack(caption_features_set_neg, axis=0)
    labels = np.stack(labels, axis=0)
    return [caption_features_set_neg, caption_features_set_pos, image_features_set], labels
