# -------------------------------------------------------------------------------
# 0) IMPORT LIBRARIES
# -------------------------------------------------------------------------------

# data transformations
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import isspmatrix
import copy

# save
import json
import os
import pickle

# visualization
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# own functions
from include.part1.triplet_loss.load_model import load_model
from include.preprocess_data import preprocessing
from include.util.util import print_progress_bar

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
transformer_triplet_loss = joblib.load(SAVE_BOW_MODEL)

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
                      transformer=None,
                      caption_embedder=caption_model_triplet,
                      min_word_length=2,
                      stem=True,
                      unique_only=False):

    #  add functionality to load caption id as a new caption
    if new_caption is None and new_caption_id is not None:
        all_captions = load_captions()
        new_caption = {new_caption_id: all_captions[new_caption_id]}
        print(f'New caption: {new_caption[new_caption_id]}')

    # preprocess caption: clean and transform to either w2v or bow
    new_caption = preprocess_caption(
        new_caption,
        transformer=transformer,
        stem=stem,
        unique_only=unique_only,
        min_word_length=min_word_length)

    # check if format is sparse
    if isspmatrix(new_caption):
        new_caption = new_caption.todense()

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
    return trans


def embed_new_image(new_image_vector=None,
                    image_embedder=None,
                    database_images=None,
                    image_id=None):

    #  add functionality to load image id as a new image
    if new_image_vector is None and image_id is not None:
        idx = np.where(database_images["id"] == image_id)[0][0]
        new_image_vector = database_images["x"][idx]
        # reshape to predict: has to be (1, F) format with F the dimensons of the embedding
        new_image_vector = (new_image_vector.reshape(1, new_image_vector.shape[0]))
    # embedding
    return image_embedder.predict(new_image_vector)


def prepare_image_database(path='include/input/image_features.csv',
                           image_embedder=None,
                           save_dir_database=None,
                           filename_database="database_images.pkl",
                           batch_size=512,
                           verbose=False):

    """
    :param path:
    :return:
    """
    if verbose:
        print("loading image features")
    database_images = pd.read_csv(path, sep=" ", header=None)
    image_ids = database_images.iloc[:, 0].values
    images_x = database_images.iloc[:, 1:].values

    if verbose:
        print("embedding image features")
    # loop to make embedding in batch sizes
    if verbose:
        print("embedding captions")
        # to print progress
        i = 0
        # needs to be greater than 1
        n = max(1, len(images_x)//batch_size)
    embedding = []
    for batch in get_batch(range(0, len(images_x)), batch_size):
        batch_embedding = image_embedder.predict(images_x[batch])
        embedding.append(batch_embedding)
        if verbose:
            i += 1
            print_progress_bar(i=i, maximum=n, post_text="Finish", n_bar=20)
    if verbose:
        print("\n")
    embedding = np.vstack(embedding)

    database_images = {
        "id": image_ids,
        "x": images_x,
        "embedding": embedding
            }
    # save
    if save_dir_database is not None:
        print("saving image database")
        if not os.path.exists(save_dir_database):
            os.makedirs(save_dir_database)
        f = open(os.path.join(save_dir_database, filename_database), "wb")
        pickle.dump(database_images, f)
    return database_images

def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def prepare_caption_database(path_raw_data="inlcude/input/results_20130124.token",
                             path_json_format="include/output/data/",
                             transformer=None,
                             caption_embedder=None,
                             clean=True,
                             stem=True,
                             min_word_length=2,
                             batch_size=1024,
                             unique_only=False,
                             save_dir_database=None,
                             filename_database="database_captions.pkl",
                             verbose=False):

    if verbose:
        print("loading captions")
    orig_captions = load_captions(path_raw_data=path_raw_data, dir_to_read_save=path_json_format)
    captions_x = copy.deepcopy(orig_captions)

    # TODO maybe also loop over preprocess_caption in batch sizes
    if verbose:
        print("preprocessing captions")
    captions_x = preprocess_caption(
        caption=captions_x,
        clean=clean,
        transformer=transformer,
        min_word_length=min_word_length,
        stem=stem,
        unique_only=unique_only
    )

    # loop to make embedding in batch sizes
    if verbose:
        print("embedding captions")
        # to print progress
        i = 0
        # needs to be greater than 1
        n = max(1, captions_x.shape[0] // batch_size)
    embedding = []
    for batch in get_batch(range(0, captions_x.shape[0]), batch_size):
        batch_X = captions_x[batch]
        if isspmatrix(batch_X):
            batch_X = batch_X.todense()
        batch_embedding = caption_embedder.predict(batch_X)
        embedding.append(batch_embedding)
        if verbose:
            i += 1
            print_progress_bar(i=i, maximum=n, post_text="Finish", n_bar=20)
    if verbose:
        print("\n")
    embedding = np.vstack(embedding)

    database_captions = {
        "original_captions": orig_captions,
        "id": np.array(list(orig_captions.keys())),
        "x": captions_x,
        "embedding": embedding
    }

    # save
    if save_dir_database is not None:
        print("saving caption database")
        if not os.path.exists(save_dir_database):
            os.makedirs(save_dir_database)
        joblib.dump(database_captions, os.path.join(save_dir_database, filename_database))
    return database_captions

def load_captions(load_from_json=False,
                          path_raw_data="include/input/results_20130124.token",
                          encoding="utf8",
                          dir_to_read_save=None):

    # check if we can read from json format
    if load_from_json and dir_to_read_save is not None:
        try:
            captions = json.load(open(os.path.join(dir_to_read_save, "all_data.json"), 'r'))
        except ValueError:
            print(f"File 'all_data.json' not found in {dir_to_read_save}")

    else:
        # open the file as read only
        file = open(path_raw_data, 'r', encoding=encoding)
        # read all text
        text = file.read()
        # close the file
        file.close()

        captions = dict()

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
            captions[caption_id] = caption_desc
        # save
        if dir_to_read_save is not None:
            if not os.path.exists(dir_to_read_save):
                os.makedirs(dir_to_read_save)
            json.dump(captions, open(os.path.join(dir_to_read_save, "all_data.json"), 'w'))

    return captions



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
    lowest_idx = np.argsort(dist)[0:k]
    # get lowest distance
    lowest_dist = dist[lowest_idx].flatten().tolist()
    # get lowest ids
    lowest_ids = database_id[lowest_idx].flatten().tolist()
    # return in dictionary format
    return dict(zip(lowest_ids, lowest_dist))

# %%
def plot_images(dic=None,
                image_dir="include/input/flickr30k-images/",
                nrows=2,
                ncols=5,
                figsize=(20, 10),
                title_fontsize=30,
                title_y=1.05,
                title_x=0.5):

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    i = 0
    for key, value in dic.items():
        img = cv2.imread(image_dir + key, cv2.COLOR_BGR2RGB)
        axes.flatten()[i].imshow(img, interpolation='bicubic')
        axes.flatten()[i].set(title=f'Distance: {round(value, 4)}')
        plt.setp(axes.flatten()[i].get_xticklabels(), visible=False)
        plt.setp(axes.flatten()[i].get_yticklabels(), visible=False)
        axes.flatten()[i].tick_params(axis='both', which='both', length=0)
        axes.flatten()[i].xaxis.grid(False)
        axes.flatten()[i].yaxis.grid(False)
        i += 1
    title = f"Ranking top {nrows * ncols} images"
    plt.suptitle(title, x=title_x, y=title_y, fontsize=title_fontsize)
    plt.tight_layout()
    plt.show()

def print_captions(dic=None,
                   database_captions=None,
                   image_id=None,
                   image_dir="include/input/flickr30k-images/",
                   figsize=(10, 8)):

    if image_id is not None:
        plt.figure(figsize=figsize)
        plt.grid(b=None)  # remove grid lines
        plt.axis('off')   # remove ticks axes
        img = cv2.imread(image_dir + image_id, cv2.COLOR_BGR2RGB)
        plt.imshow(img, interpolation='bicubic')
        plt.show()

    captions_ids = list(dic.keys())
    caption_distances = list(ranking_captions.values())
    caption_text = []
    for i, id in enumerate(captions_ids):
        text = database_captions['original_captions'][id]
        caption_text.append(text)
        print(f"Caption ID: {id}, Caption: {text}, Distance: {round(caption_distances[i],4)}) ")
    # pandas data frame format
    out = [captions_ids, caption_text, caption_distances]
    return pd.DataFrame(out, index=["caption_id", "caption_text", "distance"]).T


#%%
database_images = prepare_image_database(
    path='include/input/image_features.csv',
    image_embedder=image_model_triplet,
    save_dir_database="include/output/data/triplet_loss/database_images",
    filename_database="database_images.dat",
    batch_size=512,
    verbose=True
)
# %%
database_captions = prepare_caption_database(
    path_raw_data="include/input/results_20130124.token",
    path_json_format="include/output/data/",
    transformer=transformer_triplet_loss,
    caption_embedder=caption_model_triplet,
    save_dir_database="include/output/data/triplet_loss/database_captions",
    filename_database="database_captions.dat",
    stem=True,
    verbose=True,
    batch_size=512
)

# %%
# read in databases

# %%
database_images = joblib.load("include/output/data/triplet_loss/database_images/database_images.dat")
database_captions = joblib.load("include/output/data/triplet_loss/database_captions/database_captions.dat")

# %% new caption embedding

new_caption = {'New_Caption': 'DOG playing with a ball in the garden.!!!'}

caption = embed_new_caption(new_caption=None,
  new_caption_id="1000092795.jpg#1",
  clean=True,
  transformer=transformer_triplet_loss,
  caption_embedder=caption_model_triplet,
  min_word_length=2,
  stem=True,
  unique_only=False
)

# %%
ranking_images = rank(
    distance_metric="L2",
    new_embedding=caption,
    database_embedding=database_images['embedding'],
    database_id=database_images['id'],
    k=10
)


# %%
image = embed_new_image(new_image_vector=None,
                    image_embedder=image_model_triplet,
                    database_images=database_images,
                    image_id='10002456.jpg')

# %%
ranking_captions = rank(
    distance_metric="L2",
    new_embedding=image,
    database_embedding=database_captions['embedding'],
    database_id=database_captions['id'],
    k=10
)

# %%
out = print_captions(dic=ranking_captions,
                     database_captions=database_captions,
                     image_id='10002456.jpg',
                     image_dir="include/input/flickr30k-images/")


# %%
# TODO fix upper title
plot_images(dic=ranking_images,
            title_fontsize=30,
            figsize=(20, 10),
            title_y=1.10)
