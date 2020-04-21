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
# ------------------------------------------------
# 1) Read and process data in correct format
# ------------------------------------------------

# read in image output
images_train, images_val, images_test = preprocessing.read_split_images(path=PATH)

# %% read in caption output and split in train, validation and test set and save it
# you don't need to run this if you have already ran this before ==> see next block of code
captions_train, captions_val, captions_test = preprocessing.read_split_captions(
    path=PATH, document='results_20130124.token', encoding="utf8", dir=(BASE + "output/data"))

# %% in case you already have ran the cel above once before and don't want to run it over and over
# train
captions_train = json.load(open(BASE + 'output/data/train.json', 'r'))
# val
captions_val = json.load(open(BASE + 'output/data/val.json', 'r'))
# test
captions_test = json.load(open(BASE + 'output/data/test.json', 'r'))

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
# fit on training output (descriptions)

c_vec.fit(captions_train.values())
print(f"Size vocabulary: {len(c_vec.vocabulary_)}")
# transform on train/val/test output
captions_train_bow = [list(captions_train.keys()), c_vec.transform(captions_train.values())]
captions_val_bow = [list(captions_val.keys()), c_vec.transform(captions_val.values())]
captions_test_bow = [list(captions_test.keys()), c_vec.transform(captions_test.values())]

# %% prepare for image embedder (train/validation/test)

# 1) image data
# training data has 29783 images but we only select a subset for now
# the validation data and test data has "only" 1000 images each so we don't subset
nr_images_train = 50  # take smaller subset for experimenting
# each image has 5 captions
captions_per_image = 5

# column 0 represents the images_ids
images_train = images_train.iloc[0:nr_images_train, 1:].values
images_val = images_val.iloc[:, 1:].values
images_test = images_test.iloc[:, 1:].values

images_pairs_train = np.repeat(images_train, repeats=captions_per_image, axis=0)
images_pairs_val = np.repeat(images_val, repeats=captions_per_image, axis=0)
images_pairs_test = np.repeat(images_test, repeats=captions_per_image, axis=0)

print(f"Dimensions of the image training data: {images_pairs_train.shape}")
print(f"Dimensions of the image validation data: {images_pairs_val.shape}")
print(f"Dimensions of the image test data: {images_pairs_test.shape}")

# 2) caption data
captions_pairs_train = captions_train_bow[1][0:captions_per_image * nr_images_train, :]
captions_pairs_val = captions_val_bow[1]
captions_pairs_test = captions_test_bow[1]

print(f"Dimensions of the caption training data: {captions_pairs_train.shape}")
print(f"Dimensions of the caption validation data: {captions_pairs_val.shape}")
print(f"Dimensions of the caption test data: {captions_pairs_test.shape}")

# %%
# -------------------------------------------------
# 2) Make embedding and prepare B, F, G, S matrices
# -------------------------------------------------

# Load image and caption embedder
image_embedder = get_image_embedder(images_pairs_train.shape[1], embedding_size=C)
caption_embedder = get_caption_embedder(captions_pairs_train.shape[1], embedding_size=C)

# Create embedding matrices
# if data points (e.g. caption pairs) are supplied in a compressed format, they will be stored in compressed format
# to convert to a dense format: G.datapoints.todense()
F_train = matrix.EmbeddingMatrix(embedder=image_embedder, datapoints=images_pairs_train)
G_train = matrix.EmbeddingMatrix(embedder=caption_embedder, datapoints=captions_pairs_train)

print('F => {}'.format(F_train))
print('G => {}'.format(G_train))

# %%
# Create theta matrix
theta_train = matrix.ThetaMatrix(F_train, G_train)

# Create similarity (S) matrix of size N*M where N is the number of images and M is the number of captions
image_idx = np.repeat(np.arange(0, nr_images_train, 1), repeats=captions_per_image).tolist()
caption_idx = np.arange(0, nr_images_train * captions_per_image, 1).tolist()
image_caption_pairs = [(image_idx[i], caption_idx[i]) for i in range(len(image_idx))]

S_train = matrix.SimilarityMatrix(image_caption_pairs, nr_images_train, nr_images_train * captions_per_image)

# Create sign matrix (B)
B_train = matrix.SignMatrix(matrix_F=F_train, matrix_G=G_train, gamma=GAMMA)

print('S: {}'.format(S_train.matrix.shape))
print('Theta: {}'.format(theta_train.matrix.shape))
print('B: {}'.format(B_train.matrix.shape))
print('F: {}'.format(F_train.matrix.shape))
print('G: {}'.format(G_train.matrix.shape))

# ------------------------------------------------
# 3) Train model on training data
# ------------------------------------------------
# %%

batch_size = 32
epochs = 1
all_indices = np.arange(len(image_caption_pairs))

f_loss_sums = []
g_loss_sums = []
all_indices = permutation(all_indices)
# %%
for j in range(epochs):
    # permute batches

    for index, i in enumerate(range(0, len(image_caption_pairs), batch_size)):
        batch_interval = (i, min(len(image_caption_pairs), i + batch_size))
        # print('batch: {}'.format(batch_interval))

        # get indices of selected samples
        indices = all_indices[batch_interval[0]:batch_interval[1]]
        pairs = [image_caption_pairs[index] for index in indices]
        # print('indices -> {}'.format(indices))

        image_batch = images_train[[image_caption_pairs[index][0] for index in indices]]
        caption_batch = captions_pairs_train[[image_caption_pairs[index][1] for index in indices]]

        # make predictions for the batch
        image_embeddings = image_embedder.predict(image_batch)

        # update columns with new embeddings
        # np.put(a=F_train.matrix, ind=indices, v=image_embeddings)
        F_train.matrix[:, indices] = image_embeddings.transpose()

        theta_train.recalculate()

        # calculate loss values
        loss_f = f_loss(samples=pairs, all_pairs=image_caption_pairs,
                        theta_matrix=theta_train.matrix, F=F_train.matrix,
                        G=G_train.matrix, B=B_train.matrix, S=S_train.matrix)

        caption_embeddings = caption_embedder.predict(caption_batch.todense())
        # np.put(a=G_train.matrix, ind=indices, v=caption_embeddings)
        G_train.matrix[:, indices] = caption_embeddings.transpose()

        theta_train.recalculate()

        loss_g = g_loss(samples=pairs, all_pairs=image_caption_pairs,
                        theta_matrix=theta_train.matrix, F=F_train.matrix,
                        G=G_train.matrix, B=B_train.matrix, S=S_train.matrix)

        print('f loss ({}) -> {}'.format(loss_f.shape, loss_f[0]))
        print('g loss ({}) -> {}'.format(loss_g.shape, loss_g[0]))

        f_loss_sums.append(np.sum(loss_f[0]))
        g_loss_sums.append(np.sum(loss_g[0]))

        # update weights based on these loss values
        F_train.update_weights(loss_f)
        G_train.update_weights(loss_g)

        # recalculate sign matrix
        B_train.recalculate()
        print("batch {} done".format(index))

print(B_train)
print('samples generated')

print('f losses -> {}'.format(f_loss_sums))
plt.plot(f_loss_sums)
plt.ylabel('sums of f_loss values')
plt.show()

print('g losses -> {}'.format(g_loss_sums))
plt.plot(g_loss_sums)
plt.ylabel('sums of g_loss values')
plt.show()


# ------------------------------------------------
# 4) Test Performance
# ------------------------------------------------
# testing performance on training data
print('testing performance on training data')
trained_subset = captions_train_bow[1][0:nr_images_train * 5, :]

image_names_c = list()
image_names_i = list()
for i in range(nr_images_train):
    name = captions_train_bow[0][i * 5][: -2]
    for j in range(5):
        image_names_c.append(name)
    image_names_i.append(name)

captions_input = caption_embedder.predict(trained_subset.todense())
images_input = image_embedder.predict(images_train)

captions = [image_names_c, captions_input]
images = [image_names_i, images_input]
f_score, g_score = ranking.mean_average_precision(captions, images, captions_per_image=5)
print('performance on training data: ')
print('f_score = ' + str(round(f_score * 100, 3)) + "%")
print('g_score = ' + str(round(g_score * 100, 3)) + "%")



#%%
# --------------------------------------------------------------------
# IMPLEMENTATION BASED ON MATLAB CODE FROM THE PAPER:
# https://github.com/jiangqy/DCMH-CVPR2017/blob/master/DCMH_matlab/DCMH_matlab/process_DCMH.m
# --------------------------------------------------------------------

# 1) Read in data and preprocess
# ------------------------------

# read in image training data
images_train, _, _ = preprocessing.read_split_images(path=PATH)

# cpation training data
captions_train = json.load(open(BASE + 'output/data/train.json', 'r'))

# preprocess caption data
stemming = True
captions_train = preprocessing.clean_descriptions(
    descriptions=captions_train, min_word_length=2, stem=stemming, unique_only=False
)

# one hot encode
c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
# fit on training output (descriptions)

c_vec.fit(captions_train.values())
captions_train_bow = [list(captions_train.keys()), c_vec.transform(captions_train.values())]
#%%
# subset data
nr_images = 50
captions_per_image = 5
nr_captions = nr_pairs = nr_images*captions_per_image

images_train_subset = images_train.iloc[0:nr_images, 1:].values
caption_train_subset = captions_train_bow[1][0:nr_images*5, :]

#%%

# 2) Initialize matrices
# ------------------------------

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

max_iter = 5
batch_size = 25
loss = []

# %%
# %%

D = 250 # number of caption/images pairs
F1 = np.matmul(F, np.ones(F.shape[1], dtype=np.int8)).reshape(C, 1)  # shape: (C, 1)
i = 0
theta = 0.5 * np.matmul(F.transpose(), G)
for j in range(D):
    weight = 1/(1 + np.exp(- theta[i, j]))
    term1 = (weight * G[:, j].reshape(C, 1))
    term2 = (S[i, j] * G[:, j]).reshape(C, 1)
    term3 = term1 - term2
    term4 = 2 * GAMMA * (F[:, i].reshape(C, 1) - B[:, i].reshape(C, 1)) + 2 * ETA * F1
    dj_df = term3 + term4
    dj_df += dj_df

# %%

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