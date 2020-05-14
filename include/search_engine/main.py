# -------------------------------------------------------------------------------
# 0) IMPORT LIBRARIES
# -------------------------------------------------------------------------------

# data transformations
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import isspmatrix
import json
import os

# visualization
import seaborn as sns

# own functions
from include.part1.triplet_loss.load_model import load_model
from include.preprocess_data import preprocessing

# %% GLOBAL VARIABLES (indicated in CAPITAL letters)
sns.set()
SAVE_BOW_MODEL = 'include/output/model/triplet_loss/caption_bow_model.pkl'
MODEL_JSON_PATH = 'include/output/model/triplet_loss/best_model.json'
MODEL_WEIGHTS_PATH = 'include/output/model/triplet_loss/best_model.h5'

# -------------------------------------------------------------------------------
# 1) LOAD MODELS
# -------------------------------------------------------------------------------

"""
1.A: Load models for triplet loss (PART 1)
"""

# %% Import best trained caption and image model
print("TRIPLET LOSS: Loading models")
caption_model_triplet, image_model_triplet = load_model.load_submodels(
    model_path=MODEL_JSON_PATH, weights_path=MODEL_WEIGHTS_PATH
)

# import bow model
bow_triplet_loss = joblib.load(SAVE_BOW_MODEL)

"""
1.B: Load models (PART2) TODO add models PART 2
"""

# -------------------------------------------------------------------------------
# 2) Specify Caption/ Caption_ID or give image ID
# -------------------------------------------------------------------------------
# %%

# -------------------------------------------------------------------------------
# HELPER FUNCTIONS (NOT FINISHED)
# -------------------------------------------------------------------------------

def embed_new_caption(new_caption=None,
                      new_caption_id=None,
                      clean=True,
                      transformer=bow_triplet_loss,
                      caption_embedder=caption_model_triplet,
                      min_word_length=2,
                      stem=True,
                      unique_only=False):

    # TODO: add functionality to load caption id as a new caption
    if new_caption is None and new_caption_id is not None:
        all_captions = load_captions()
        new_caption = {new_caption_id: all_captions[new_caption_id]}

    # preprocess caption: clean and transform to either w2v or bow
    new_caption = preprocess_caption(
        new_caption,
        transformer=transformer,
        stem=stem,
        unique_only=unique_only,
        min_word_length=min_word_length)

    return caption_embedder(new_caption)


def preprocess_caption(caption=None,
                       clean=True,
                       transformer=None,
                       min_word_length=2,
                       stem=True,
                       unique_only=False):
    if clean:
        caption = preprocessing.clean_descriptions(
            descriptions=caption,
            min_word_length=min_word_length,
            stem=stem,
            unique_only=unique_only,
            verbose=False
        )

    # convert caption to either bow or word2vec
    trans = transformer.transform(caption)
    if isspmatrix(trans):
        trans = trans.todense()
    return trans


def embed_new_image(new_image=None, image_embedder=None):

    # TODO: add functionality to load image id as a new image
    # embedding
    return image_embedder.predict(new_image)


def prepare_image_database(path='include/input/image_features.csv',
                           image_embedder=None,
                           save_dir=None):

    """
    :param path:
    :return:
    """
    database_images = pd.read_csv(path, sep=" ", header=None)
    image_ids = database_images.iloc[:, 0].values
    images = database_images.iloc[:, 1:].values
    embedding = image_embedder.predict(images)
    database_images = {
        "ID": image_ids,
        "X": images,
        "Embedding": embedding
            }
    # save
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json.dump(database_images, open(os.path.join(save_dir, "database_images.json"), 'w'))
    return database_images


def prepare_caption_database(dir="include/output/data/",
                             transformer=None,
                             caption_embedder=None,
                             clean=True,
                             stem=True,
                             min_word_length=2,
                             unique_only=False,
                             save_dir=None):

    captions = load_captions(dir_to_read_save=dir)
    captions_X = preprocess_caption(
        caption=captions,
        clean=clean,
        transformer=transformer,
        min_word_length=min_word_length,
        stem=stem,
        unique_only=unique_only
    )
    embedding = caption_embedder.predict(captions_X)
    database_captions = {
        "ID": np.array(captions.values()),
        "X": captions_X,
        "Embedding": embedding
    }
    # save
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json.dump(database_captions, open(os.path.join(save_dir, "database_captions.json"), 'w'))
    return database_captions

def load_captions(load_from_json=False,
                          path_raw_data="include/input/results_20130124.token",
                          encoding="utf8",
                          dir_to_read_save=None):

    # check if we can read from json format
    if load_from_json and dir_to_read_save is not None:
        try:
            database_captions = json.load(open(os.path.join(dir_to_read_save, "all_data.json"), 'r'))
        except ValueError:
            print(f"File 'all_data.json' not found in {dir_to_read_save}")

    else:
        # open the file as read only
        file = open(path_raw_data, 'r', encoding=encoding)
        # read all text
        text = file.read()
        # close the file
        file.close()

        database_captions = dict()

        # process lines
        for line in text.split('\n'):
            # split line by white space
            tokens = line.split()
            # ignore lines shorter than two
            if len(line) < 2:
                continue
            # take the first token as the image id, the rest as the description
            caption_id, caption_desc = tokens[0], tokens[1:]
            # convert description tokens back to string
            caption_desc = ' '.join(caption_desc)
            database_captions[caption_id] = caption_desc
        # save
        if dir_to_read_save is not None:
            if not os.path.exists(dir_to_read_save):
                os.makedirs(dir_to_read_save)
            json.dump(database_captions, open(os.path.join(dir_to_read_save, "all_data.json"), 'w'))

    return database_captions



def rank(distance_metric="L2",
         new_embedding=None,
         database_embedding=None,
         database_id=None,
         k=10):

    assert distance_metric in ["L2", "Hamming"]

    if distance_metric == "L2":
        dist = norm(database_embedding - new_embedding, ord=2, axis=1)
    else:
        pass  # TODO: add hamming distance

    # convert to numpy array
    # find lowest indices
    lowest_idx = np.argpartition(dist, kth=range(len(dist)))[0:k]
    # get lowest distance
    lowest_dist = dist[lowest_idx].flatten().tolist()
    # get lowest ids
    lowest_ids = database_id[lowest_idx].flatten().tolist()
    # return in dictionary format
    return dict(zip(lowest_ids, lowest_dist))



def plot_images(dic):
    pass  # TODO plot top K images

def print_captions(dic):
    pass  # TODO print top K captions




