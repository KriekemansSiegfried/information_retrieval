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
        all_captions = load_caption_database()
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


#%%

def embed_new_image(new_image=None, image_embedder=None):

    # TODO: add functionality to load image id as a new image
    # embedding
    return image_embedder.predict(new_image)

# %%
def load_image_database(path='include/input/image_features.csv'):
    """

    :param path:
    :return:
    """
    database_images = pd.read_csv(path, sep=" ", header=None)
    image_ids = database_images.iloc[:, 0].values
    images = database_images.iloc[:, 1:].values
    return {
        "ID": image_ids,
        "X": images
            }

# %%

def load_caption_database(load_from_json=False,
                          path_raw_data="include/input/results_20130124.token",
                          encoding="utf8",
                          dir_to_read_save="include/output/data/"):

    # check if we can read from json format
    if load_from_json and dir_to_read_save is not None:
        database_captions = json.load(open(dir_to_read_save, 'r'))

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
        if dir_to_read_save is not None:
            json.dump(database_captions, open(os.path.join(dir_to_read_save, "all_data.json"), 'w'))

    return database_captions


def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def rank(distance_metric="L2",
         new_embedding=None,
         database_X=None,
         database_id=None,
         database_model=None,
         batch_size=512,
         k=10):

    #assert distance_metric in ["L2", "Hamming"]
    store_dist = []
    # loop over in batches (predicting the whole database it to large)
    for batch in get_batch(list(range(0, len(database_X))), batch_size):
        database_embedding = database_model(database_X[batch, :])
        if distance_metric == "L2":
            dist = norm(database_embedding - new_embedding, ord=2, axis=1)
        else:
            pass  # TODO: add hamming distance

        # store distance
        store_dist += dist.flatten().tolist()

    # convert to numpy array
    store_dist = np.array(store_dist)
    # find lowest indices
    lowest_idx = np.argpartition(store_dist, kth=range(len(store_dist)))[0:k]
    # get lowest distance
    lowest_dist = store_dist[lowest_idx].flatten().tolist()
    # get lowest ids
    lowest_ids = database_id[lowest_idx].flatten().tolist()
    # return in dictionary format
    return dict(zip(lowest_ids, lowest_dist))
# %%



# %%
def plot_images(dic):
    pass

def print_captions(dic):
    pass


# %% Test

# %%
caption_test = embed_new_caption(
    new_caption_id='1067675215.jpg#4',
    caption_embedder=caption_model_triplet,
    transformer=bow_triplet_loss
)

# %%
database_images = load_image_database()

# %%

# rank images
ranking_out = rank(
     new_embedding=caption_test,
     database_X=database_images['X'],
     database_id=database_images['ID'],
     database_model=image_model_triplet,
     batch_size=512,
     k=10)

# %% TODO rank captions
databse_captions = load_caption_database()
ID_captions = databse_captions.keys()
#%%
databse_captions = preprocess_caption(
    caption=databse_captions,
    clean=True,
    transformer=bow_triplet_loss,
    min_word_length=2,
    stem=True,
    unique_only=False
)
# %%
ranking_out = rank(
     new_embedding=caption_test,
     database_X=databse_captions,
     database_id=np.array(ID_captions),
     database_model=image_model_triplet,
     batch_size=512,
     k=10)