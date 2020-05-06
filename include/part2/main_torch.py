# data pre processing
import json
from math import ceil

import numpy as np
# visualization
import seaborn as sns
import torch
from torch.autograd import Variable
from torch.optim import SGD

from include.part2.loss.loss import f_loss_torch
from include.part2.torch_embedding.embedder import Embedder
from numpy.random import permutation
from sklearn.feature_extraction.text import CountVectorizer

from include.part2 import ranking
from include.part2.embedding import embedding
from include.part2.embedding import matrix
# own functions
from include.preprocess_data import preprocessing

# %% GLOBAL VARIABLES (indicated in CAPITAL letters)
# BASE = "include/"
BASE = ""
PATH = BASE + "input/"

sns.set()


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    return label1.matmul(label2.transpose(0, 1)) > 0


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
# TODO:  - Verify whether gradients are computed correctly, if so make a more efficient implementation (vectorized)
#        - Backpropgation
#        - Implement Loss function
#        - make it work on new captions (to showcase) --> IS PART OF SEARCH ENGINE
#        - look at efficiency and fine-tuning (partially done)
#        - fix multiple captions per images and vice versa (see footnote assingment)
#        -
#        - DONE: perform map10 on testing data
#        - DONE: perform map10 on testing data


# %%
# ------------------------------
# 1) Read in data and preprocess
# ------------------------------

# read in image output
images_train, images_val, images_test = preprocessing.read_split_images(path=PATH)

# %% read in caption output and split in train, validation and test set and save it
# you don't need to run this if you have already ran this before ==> see next block of code
# captions_train, captions_val, captions_test = preprocessing.read_split_captions(
#     path=PATH, document='results_20130124.token', encoding="utf8", dir=(BASE + "output/data"))

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
# captions_train = preprocessing.clean_descriptions(
#     descriptions=captions_train, min_word_length=2, stem=stemming, unique_only=False
# )
# captions_val = preprocessing.clean_descriptions(
#     descriptions=captions_val, min_word_length=2, stem=stemming, unique_only=False
# )
# captions_test = preprocessing.clean_descriptions(
#     descriptions=captions_test, min_word_length=2, stem=stemming, unique_only=False
# )

# %% convert to bow
c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
# fit on training output (descriptions)

c_vec.fit(captions_train.values())
print(f"Size vocabulary: {len(c_vec.vocabulary_)}")
# transform on train/val/test output
captions_train_bow = [list(captions_train.keys()), c_vec.transform(captions_train.values())]
captions_val_bow = [list(captions_val.keys()), c_vec.transform(captions_val.values())]
captions_test_bow = [list(captions_test.keys()), c_vec.transform(captions_test.values())]

# %%
# ----------------------------------
# 2) Initialize matrices: F, G, B, S
# ----------------------------------

nr_images = 50
captions_per_image = 5
nr_captions = nr_pairs = nr_images * captions_per_image

# Create similarity (S) matrix of shape (N*M) where N is the number of images and M is the number of captions
image_idx = np.repeat(np.arange(0, nr_images, 1), repeats=captions_per_image).tolist()
caption_idx = np.arange(0, nr_captions, 1).tolist()
image_caption_pairs = np.array([(image_idx[i], caption_idx[i]) for i in range(len(image_idx))])
# Define torch variable containing similarity information of captions and images
S = Variable(torch.from_numpy(matrix.SimilarityMatrix(image_caption_pairs, nr_images, nr_captions).matrix))

# F and G: shape (C, D) where D is the number of image_caption pairs

F_buffer = torch.randn(nr_captions, C)
G_buffer = torch.randn(nr_captions, C)

# initialize B: shape: (C,D)
B = torch.sign(F_buffer + G_buffer)

image_train_subset = np.repeat(a=images_train.iloc[0:nr_images, 1:].values, repeats=5, axis=0)
caption_train_subset = captions_train_bow[1][0:nr_captions, :]
# networks
image_embedder = Embedder(image_train_subset.shape[1], C)
caption_embedder = Embedder(caption_train_subset.shape[1], C)

# TODO: optimize lr value
image_optimizer = SGD(image_embedder.parameters(), lr=0.001)
caption_optimizer = SGD(caption_embedder.parameters(), lr=0.001)

batch_size = 25
epochs = 5
data_size = nr_images * captions_per_image

ones_batch = torch.ones(batch_size)
ones_other = torch.ones(data_size - batch_size)

for epoch in range(epochs):

    # image update loop
    for i in range(nr_images // batch_size):
        # get random set of indices as batch
        batch_indices = np.random.permutation(data_size)[:batch_size]
        print('batch_indices : {}'.format(batch_indices))
        other_indices = np.setdiff1d(range(data_size), batch_indices)

        batch_images = Variable(torch.from_numpy(image_train_subset[batch_indices, :]))
        batch_labels = S[batch_indices, :]

        # TODO: is this correct?
        sim = Variable(S[batch_indices, :])

        # calc current calculated similarity
        batch_image_embedding = image_embedder(batch_images)
        F_buffer[batch_indices, :] = batch_image_embedding  # update F matrix with new embeddings

        F = Variable(F_buffer)
        G = Variable(G_buffer)
        B_var = Variable(B)
        ones_batch_var = Variable(ones_batch)
        ones_other_var = Variable(ones_other)

        theta_x = 1.0 / 2 * torch.matmul(G,F.t())

        image_gradients = f_loss_torch(batch_indices, image_caption_pairs, theta_x, F, G, B, S)
        image_optimizer.zero_grad()
        print('backprop')
        batch_image_embedding.backward(image_gradients)
        image_optimizer.step()

    for i in range(data_size // batch_size):
        # get random set of indices as batch
        batch_indices = np.random.permutation(data_size)[:batch_size]
        other_indices = np.setdiff1d(range(data_size), batch_indices)

        batch_captions = Variable(torch.from_numpy(caption_train_subset.todense()[batch_indices, :]))
        batch_labels = image_caption_pairs[batch_indices, :]

        # calc current calculated similarity
        sim = S[batch_indices, :]

        batch_text_embedding = caption_embedder(batch_captions)
        G_buffer[batch_indices, :] = batch_text_embedding
        F = Variable(F_buffer)
        G = Variable(G_buffer)

        theta_y = 1.0 / 2 * torch.matmul(batch_text_embedding, F.t())
        logloss_y = -torch.sum(sim * theta_y - torch.log(1.0 + torch.exp(theta_y)))
        quantization_y = torch.sum(torch.pow(B[batch_indices, :] - batch_text_embedding, 2))
        balance_y = torch.sum(torch.pow(
            torch.matmul(batch_text_embedding.t(), ones_batch) + torch.matmul(G[other_indices].t(), ones_other), 2))
        loss_y = logloss_y + GAMMA * quantization_y + ETA * balance_y
        loss_y /= (batch_size * data_size)

        caption_optimizer.zero_grad()
        loss_y.backward()
        caption_optimizer.step()

    # update sign matrix
    B = torch.sign(F_buffer + G_buffer)

# ------------------------------------------------
# 4) Test Performance
# ------------------------------------------------
# testing performance on training data
print('testing performance on training data')

image_names_c = list()
image_names_i = list()
for i in range(nr_images):
    name = captions_test_bow[0][i * captions_per_image][: -2]
for j in range(captions_per_image):
    image_names_c.append(name)
image_names_i.append(name)

captions_input = caption_embedder.predict(caption_train_subset.todense())
images_input = image_embedder.predict(image_train_subset)

captions = [image_names_c, captions_input]
images = [image_names_i, images_input]
f_score, g_score = ranking.mean_average_precision(captions, images, captions_per_image=captions_per_image)
print('performance on training data: ')
print('f_score = ' + str(round(f_score * 100, 3)) + "%")
print('g_score = ' + str(round(g_score * 100, 3)) + "%")
