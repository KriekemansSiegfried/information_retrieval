# data pre processing
import numpy as np
from include.part2 import ranking
from numpy.random import  permutation
from sklearn.feature_extraction.text import CountVectorizer
import json
from math import floor

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# own functions
from include.preprocess_data import preprocessing
from include.part2.embedding.embedding import get_caption_embedder, get_image_embedder
from include.part2.embedding import matrix
from include.part2.loss.loss import f_loss, g_loss

# %% GLOBAL VARIABLES (indicated in CAPITAL letters)
# BASE = "include/"
BASE = "include/"
PATH = BASE + "input/"

sns.set()

"""
Parameters:
"""
GAMMA = 1
ETA = 1
C = 32
"""
Training loop:

1. Create neural networks for image and caption embedding
2. initialize B as ysign(F + G)
3. update weights of F based on custom loss
4. update weights of G based on custom loss
5. update B as ysign(F + G)

"""
# %%
# ------------------------------
# 1) Read in data and preprocess
# ------------------------------

# read in image model
images_train, images_val, images_test = preprocessing.read_split_images(path=PATH)

# %% read in caption model and split in train, validation and test set and save it
# you don't need to run this if you have already ran this before ==> see next block of code
captions_train, captions_val, captions_test = preprocessing.read_split_captions(
    path=PATH, document='results_20130124.token', encoding="utf8", dir=(BASE + "model/data"))

# %% in case you already have ran the cel above once before and don't want to run it over and over
# train
captions_train = json.load(open(BASE + 'model/data/train.json', 'r'))
# val
captions_val = json.load(open(BASE + 'model/data/val.json', 'r'))
# test
captions_test = json.load(open(BASE + 'model/data/test.json', 'r'))

# %% clean captions (don't run this more than once or
# you will prune your caption dictionary even further as it has the same variable name)

# experiment with it: my experience: seems to work better if you apply stemming when training
stemming = True
captions_train = preprocessing.clean_descriptions(
    descriptions=captions_train, min_word_length=2, stem=stemming, unique_only=False
)
captions_val = preprocessing.clean_descriptions(
    descriptions=captions_val, min_word_length=2, stem=stemming, unique_only=False
)
captions_test = preprocessing.clean_descriptions(
    descriptions=captions_test, min_word_length=2, stem=stemming, unique_only=False
)

# %% convert to bow
c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
# fit on training model (descriptions)

c_vec.fit(captions_train.values())
print(f"Size vocabulary: {len(c_vec.vocabulary_)}")
# transform on train/val/test model
captions_train_bow = [list(captions_train.keys()), c_vec.transform(captions_train.values())]
captions_val_bow = [list(captions_val.keys()), c_vec.transform(captions_val.values())]
captions_test_bow = [list(captions_test.keys()), c_vec.transform(captions_test.values())]



#%%
# ----------------------------------
# 2) Initialize matrices: F, G, B, S
# ----------------------------------

nr_images = 50
captions_per_image = 5
nr_captions = nr_pairs = nr_images*captions_per_image

images_train_subset = images_train.iloc[0:nr_images, 1:].values
caption_train_subset = captions_train_bow[1][0:nr_captions, :]


# F and G: shape (C, D) where D is the number of image_caption pairs
C = 32
F = np.random.normal(0, 1, C*nr_pairs).reshape(C, nr_pairs)
G = np.random.normal(0, 1, C*nr_pairs).reshape(C, nr_pairs)

# Create similarity (S) matrix of shape (N*M) where N is the number of images and M is the number of captions
image_idx = np.repeat(np.arange(0, nr_images, 1), repeats=captions_per_image).tolist()
caption_idx = np.arange(0, nr_captions, 1).tolist()
image_caption_pairs = [(image_idx[i], caption_idx[i]) for i in range(len(image_idx))]
S = matrix.SimilarityMatrix(image_caption_pairs, nr_images, nr_captions).matrix

# initialize B: shape: (C,D)
B = GAMMA * np.sign(F + G)

# networks
image_embedder = get_image_embedder(images_train_subset.shape[1], embedding_size=C)
caption_embedder = get_caption_embedder(caption_train_subset.shape[1], embedding_size=C)

epochs = 5
N_x = 25  # batchsize
N_y = 25  # batchsize
loss = []


# %%
# --------------------------------------------------------------------
# IMPLEMENTATION BASED ON MATLAB CODE FROM THE PAPER (Vectorized code)
# !! CODE IS NOT WORKING !!
# https://github.com/jiangqy/DCMH-CVPR2017/blob/master/DCMH_matlab/DCMH_matlab/process_DCMH.m
# --------------------------------------------------------------------

# 1) Read in data and preprocess
# ------------------------------

# %%
# subset data
nr_images = 50
captions_per_image = 5
nr_captions = nr_pairs = nr_images * captions_per_image

images_train_subset = images_train.iloc[0:nr_images, 1:].values
caption_train_subset = captions_train_bow[1][0:nr_images * 5, :]

# %%

# 2) Initialize matrices
# ------------------------------

# F and G: shape (C, D) where D is the number of image_caption pairs
C = 32
F = np.random.normal(0, 1, C * nr_pairs).reshape(C, nr_pairs)
G = np.random.normal(0, 1, C * nr_pairs).reshape(C, nr_pairs)

# Create similarity (S) matrix of shape (N*M) where N is the number of images and M is the number of captions

image_idx = np.repeat(np.arange(0, nr_images, 1), repeats=captions_per_image).tolist()
caption_idx = np.arange(0, nr_captions, 1).tolist()
image_caption_pairs = [(image_idx[i], caption_idx[i]) for i in range(len(image_idx))]
S = matrix.SimilarityMatrix(image_caption_pairs, nr_images, nr_captions).matrix

# initialize B: shape: (C,D)
B = GAMMA * np.sign(F + G)

# networks
image_embedder = get_image_embedder(images_train_subset.shape[1], embedding_size=C)
caption_embedder = get_caption_embedder(caption_train_subset.shape[1], embedding_size=C)

max_iter = 5
batch_size = 25
loss = []

# %%

"""

# 3) Train model
# ------------------------------

for epoch in range(max_iter):

    print(epoch)

    # 1) loop for images (X)
    for b in range(floor(nr_images/batch_size)):

        # Randomize indices
        R = np.random.choice(nr_images, nr_images, replace=False)
        # Select T indices:
        T = R[0:batch_size]
        X = images_train_subset[T, :]
        # For each sampled point xi calculate F*i
        output_F = image_embedder(X).numpy()  # shape: (T, C)
        # update
        F[:, T] = output_F.transpose()  # shape: (C, T)

        # calculate the gradient (vectorized implementation assignment)
        #  - F and G are of shape: (C, D)
        #  - S is of shape (N, M), with N the nr images and M is the nr captions = image_caption_pairs = D

        F1 = np.matmul(F, np.ones(F.shape[1], dtype=np.int8)).reshape(C, 1)  # shape: (C, 1)
        theta = 0.5 * np.matmul(F.transpose(), G)  # shape: (D, D)
        weight = 1 / (1 + np.exp(-theta[T, :]))  # shape: (T, D)
        part1 = np.matmul(G, weight.transpose())  # shape: (C, T)
        part2 = np.matmul(G, S[T, :].transpose())  # shape: (C, T)
        term1 = 0.5 * (part1 - part2)  # shape: (C, T)  (logloss)
        term2 = 2 * GAMMA * (F[:, T] - B[:, T]) + 2 * ETA * F1  # shape: (C, T)
        dj_df_image = term1 + term2  # shape: (C, T)

        # update parameters theta_X by using backprop


    # 2) loop for captions (Y)
    for b in range(floor(nr_captions/batch_size)):

        # Randomize indices
        R = np.random.choice(nr_captions, nr_captions, replace=False)
        # Select T indices:
        T = R[0:batch_size]
        Y = caption_train_subset[T, :]
        # For each sampled point xj calculate G*j
        output_G = caption_embedder(Y.todense()).numpy()  # shape: (T, C)
        # update
        G[:, T] = output_G.transpose()  # shape: (C, T)

        # calculate the gradient (vectorized implementation assignment)
        #  - F and G are of shape: (C, D)
        #  - S is of shape (N, M), with N the nr images and M is the nr captions = image_caption_pairs = D

        G1 = np.matmul(G, np.ones(G.shape[1], dtype=np.int8)).reshape(C, 1)  # shape: (C, 1)
        theta = 0.5 * np.matmul(F.transpose(), G)  # shape: (D, D)
        weight = 1 / (1 + np.exp(-theta[:, T]))  # shape: (D, T)
        part1 = np.matmul(F, weight)  # shape: (C, T)
        part2 = np.matmul(S[:, T], F)  # shape: should be (C, T), but not possible
        term1 = 0.5 * (part1 - part2)  # shape: (C, T)  (logloss)
        term2 = 2 * GAMMA * (F[:, T] - B[:, T]) + 2 * ETA * G1  # shape: (C, T)
        dj_dg_text = term1 + term2  # shape: (C, T)

        # update parameters theta_Y by using backprop


    # 3) update B
    B = GAMMA * np.sign(F + G)

    # 4) calculate loss

"""