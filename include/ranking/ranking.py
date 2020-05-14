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
                   distance_metric="L2",
                   add_correct_id=True,
                   verbose=True):
    """
    Computes the ranking of either the captions or images

    :param caption_embed: Numpy array, embedding (predictions) of the captions
    :param caption_id: Numpy array, contains the captions id's
    :param image_embed: Numpy array, embedding (predictions) of the images
    :param image_id: Numpy array, contains the image id's
    :param retrieve: String, either "captions" or "images". Default "captions"
    :param k: Integer, determines how many captions_ids/image_ids to rank. Default is True
    :param distance_metric: String, computes the distance. Either "L2" or "Hamming". Default is "L2"
    :param add_correct_id: Boolean, add ranking(s) of the correct caption/image(s). Default is True
    :param verbose: Boolean, print progress. Default is True
    :return: Dictionary, contains the ranking with the distances and (optionally) correct_ids
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
        if distance_metric == "L2":
            dist = norm(database_features - new_embedding_features[i], ord=2, axis=1)
        elif distance_metric == "Hamming":
            dist = 1 - np.mean((database_features - new_embedding_features[i] == 0), axis=1)
        else:
            print("Choose a listed distance metric: available distance metrics include L2 and Hamming distance")

        # get indices of distances (dist) from low to high
        rank_all = np.argpartition(dist, kth=range(len(dist)))
        # get distances (low to high)
        dist_all = dist[rank_all].tolist()
        # get 10 lowest distances
        dist_k = dist_all[0:k]
        # get image idx of rank
        if retrieve == "captions":
            ids_k = database_id_original[rank_all[0:k]].tolist()
        elif retrieve == "images":
            ids_k = database_id[rank_all[0:k]].tolist()
        else:
            print("error, retrieve should be image or caption")
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


def average_precision(dic=None, gtp=1):
    """
    Computes the average precision for each caption_id/ image_id

    :param dic:, (Nested) Dictionary, contains the ranking with the distances
    :param gtp: Integer, the number of ground truth positives. Default is 1. For captions this should be 5
    :return: Pandas DataFrame, containing the average precision for each caption_id/ image_id
    """

    # store average precision
    store_ap = {}
    print(f"The number of ground true positives is {gtp} when computing the average precision")
    for key, value in dic.items():

        # get the id's of the ranked images/captions
        list_ranking = [item.split(".")[0] for item in value[0]]

        # check if the correct_id'(s) is (are) in the "list_ranking"
        if key.split(".")[0] in list_ranking:
            ap = []
            correct = 0
            # check the place of the caption_id/image_id in the ranking
            # see article page 5 for exact reasoning:
            # https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
            for i, k in enumerate(list_ranking):
                if k == key.split(".")[0]:
                    correct += 1
                    ap.append(correct / (i + 1))
                else:
                    ap.append(0)

            n = 0
            for x in range(gtp):
                n = n + 1/(x+1)

            store_ap[key] = 1 / n * sum(ap)
        else:
            store_ap[key] = 0

    return pd.DataFrame.from_dict(store_ap, orient='index', columns=['average_precision'])