import numpy as np
import pandas as pd
from numpy.linalg import norm

from include.util.util import print_progress_bar


def rank_embedding(caption_embed=None,
                   caption_id=None,
                   image_embed=None,
                   image_id=None,
                   retrieve="captions",
                   k=10,
                   add_correct_id=True,
                   verbose=True):
    """

    :param caption_embed:
    :param caption_id:
    :param image_embed:
    :param image_id:
    :param retrieve:
    :param k:
    :param add_correct_id:
    :param verbose:
    :return:
    """

    if retrieve == "captions":
        new_embedding_id = image_id
        new_embedding_features = image_embed
        database_id_original = caption_id
        database_id = pd.Series(database_id_original).str.split(".").str[0].values
        database_features = caption_embed
    elif retrieve == "images":
        new_embedding_id = caption_id
        new_embedding_features = caption_embed
        database_id = image_id
        database_features = image_embed
    else:
        print("error, retrieve should be image or caption")

    ranking = {}
    for i, key in enumerate(new_embedding_id):

        # compute distances
        dist = norm(database_features - new_embedding_features[i], ord=2, axis=1)
        # rank indexes small to large
        rank_all = np.argpartition(dist, kth=range(len(dist)))
        dist_all = dist[rank_all].tolist()
        # get 10 lowest distances
        dist_k = dist_all[0:k]
        # get image idx of rank
        if retrieve == "captions":
            ids_k = database_id_original[rank_all[0:k]].tolist()
        else:
            ids_k = database_id[rank_all[0:k]].tolist()
        # get correct idx and distance
        if add_correct_id:
            idx = list(np.where(key.split(".")[0] == database_id)[0])
            correct_idx = [np.where(j == rank_all)[0][0] for j in idx]
            dist_correct_idx = np.array(dist_all)[correct_idx].tolist()
            # store in dictionary
            ranking[key] = (dict(zip(ids_k, dist_k)), dict(zip(correct_idx, dist_correct_idx)))
        # store in dictionary
        else:
            ranking[key] = dict(zip(ids_k, dist_k))

        # print progress
        if verbose:
            print_progress_bar(i=i, maximum=len(new_embedding_id), post_text="Finish", n_bar=20)
    return ranking


def average_precision(dic, gtp=1):
    """

    :param dic:
    :param gtp:
    :return:
    """

    # store average precision
    store_ap = {}
    print(f"The number of ground true positives is {gtp} when computing the average precision")
    for key, value in dic.items():

        list_ranking = [item.split(".")[0] for item in value[0]]

        if key.split(".")[0] in list_ranking:
            ap = []
            correct = 0
            for i, k in enumerate(list_ranking):
                if k == key.split(".")[0]:
                    correct += 1
                    ap.append(correct / (i + 1))
                else:
                    ap.append(0)

            store_ap[key] = 1 / gtp * sum(ap)
        else:
            store_ap[key] = 0

    return pd.DataFrame.from_dict(store_ap, orient='index', columns=['average_precision'])
