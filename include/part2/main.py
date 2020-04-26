# data pre processing
import numpy as np
from include.part2 import ranking
from numpy.random import permutation
from sklearn.feature_extraction.text import CountVectorizer
import json
from math import ceil

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# own functions
from include.preprocess_data import preprocessing
from include.part2.embedding import embedding
from include.part2.embedding import matrix
from include.part2.loss.loss import f_loss, g_loss

# %% GLOBAL VARIABLES (indicated in CAPITAL letters)
# BASE = "include/"
BASE = "../"
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
S_dummi = matrix.SimilarityMatrix(image_caption_pairs, nr_images, nr_captions).matrix
S = np.zeros((nr_captions, nr_captions))
for j in range(nr_captions):
    S[j, :] = S_dummi[int(j/captions_per_image), :]

# initialize B: shape: (C,D)
B = GAMMA * np.sign(F + G)

# networks
image_embedder = embedding.get_image_embedder(images_train_subset.shape[1], embedding_size=C)
caption_embedder = embedding.get_caption_embedder(caption_train_subset.shape[1], embedding_size=C)

epochs = 5
N_x = 25  # batchsize
N_y = 25  # batchsize
loss = []

# %%
# -------------------------------------------------------------------------------------------
# 3) Train model:  I very closely followed the algorithm as outlined in the paper on page 3236
#    NOTE: this is not efficient code, but let's first focus on getting
#    this right before looking at efficiency
# -------------------------------------------------------------------------------------------
origional_captions_input = caption_embedder.predict(caption_train_subset.todense())
origional_images_input = image_embedder.predict(images_train_subset)

for epoch in range(epochs):
    print(epoch)
    # 1) loop for images (X)
    for b in range(ceil(nr_images / N_x)):

        # prev_weights = image_embedder.get_weights()

        # Randomize indices
        R = np.random.choice(nr_images, nr_images, replace=False)
        # Randomly sample N_x points from X to construct mminibatch
        T = R[0:N_x]
        mini_batch_X = images_train_subset[T, :]
        # 1.1) For each sampled point xi in the minibatch calculate F*i by forward propagation
        # 1.2) calculate the derivative according to 3
        dj_df_T = np.zeros((C, 1))
        # F1 = sum of the columns of F = F x (np.ones((F.shape[1], 1)) (Note: our F is transpose(F))
        F1 = F.sum(axis=1).reshape(C, 1)
        # theta = 0.5 * np.matmul(F.transpose(), G)
        theta = 0.5 * np.matmul(G.transpose(), F)
        for index_mb, i in enumerate(T):
            # embedding_F = prediction of the network for mini_batch_X[index_mb, :], shape is (1, C)
            embedding_F = image_embedder(mini_batch_X[index_mb, :].reshape(1, mini_batch_X.shape[1])).numpy()
            # only use flatten if F[:,i] is of shape C, 1
            F[:, i] = embedding_F.transpose().flatten()  # F shape: (C, T)
            # initialize matrix to store gradients shape: (C, 1)
            """--new version--"""
            # 0.5*SUM((sigma[i,j] - S[i,j])*G[:,j]) + 2*gamma*(F[:,i]-B[:,i]) + 2*eta*F1
            # <=> 0.5*(sigma[i,:] - S[i,:])*G + term0  (thus term0 = 2*gamma*(F[:,i]-B[:,i]) + 2*eta*F1)
            #                                          (the was sum also removed and performed on all columns in one go)
            # <=> 0.5*s_weights*G + term0              (thus s_weights = (sigma[i,:] - S[i,:])
            #                                          (and sigma[i,:] = 1/(1+exp(theta[i,:])))
            term0 = 2 * GAMMA * (F[:, i].reshape(C, 1) - B[:, i].reshape(C, 1)) + 2 * ETA * F1
            # s_weights = (1 / (1 + np.exp(- theta[i, :]))) - S[i, :]
            s_weights = (1 / (1 + np.exp(- theta[:, i]))) - S[i, :]
            total = 0.5*s_weights*G + term0
            dj_df_D = total.sum(axis=1).reshape(C, 1)
            dj_df_T = dj_df_T + dj_df_D
            """--new version--"""
            """--old version--
            dj_df_D = np.zeros((C, 1))
            # inner loop: i to D (I assume D is of length: nr_captions or nr_pairs)
            term0 = 2 * GAMMA * (F[:, i].reshape(C, 1) - B[:, i].reshape(C, 1)) + 2 * ETA * F1
            for j in range(nr_pairs):
                weight = 1 / (1 + np.exp(- theta[i, j]))
                term1 = weight * G[:, j]
                term2 = (S[i, j] * G[:, j])
                term3 = (term1 - term2).reshape(C, 1)
                total = term3 + term0
                dj_df_D = dj_df_D + total
            dj_df_T = dj_df_T + 0.5 * dj_df_D
               --old version--"""
        # 1.3) update the parameters theta_X by using back propagation
        new_weights = embedding.backprop_weights(image_embedder, dj_df_T)
        image_embedder.set_weights(new_weights)
        # new_weights = image_embedder.get_weights()
        # print('difference: ')
        # for i in range(len(new_weights)):
        #     print(new_weights[i] - prev_weights[i])



    # 2) loop for images (captions)
    for b in range(ceil(nr_captions / N_y)):
        # Randomize indices
        R = np.random.choice(nr_captions, nr_captions, replace=False)
        # Randomly sample N_x points from X to construct minibatch
        T = R[0:N_y]
        mini_batch_Y = caption_train_subset[T, :].todense()
        # 2.1) For each sampled point xi in the minibatch calculate F*i by forward propagation
        # 2.2) calculate the derivative according to 3
        dj_dg_T = np.zeros((C, 1))
        # G1 = np.matmul(G, np.ones(G.shape[1], dtype=np.int8)).reshape(C, 1)           # <-- now done by the line below
        G1 = G.sum(axis=1).reshape(C, 1)
        for index_mb, j in enumerate(T):
            # embedding_G, shape: (1, C)
            embedding_G = caption_embedder(mini_batch_Y[index_mb, :].reshape(1, mini_batch_Y.shape[1])).numpy()
            # only use flatten if  G[: i] is of shape C, 1
            G[:, j] = embedding_G.transpose().flatten()  # G shape: (C, T)
            # initialize matrix to store gradients shape: (C, 1)
            """--new version--"""
            # G1 = G.sum(axis=1).reshape(C, 1)
            term0 = 2 * GAMMA * (G[:, j].reshape(C, 1) - B[:, j].reshape(C, 1)) + 2 * ETA * G1
            s_weights = (1 / (1 + np.exp(- theta[:, j]))) - S[j, :]
            total = 0.5*s_weights*F + term0
            dj_dg_D = total.sum(axis=1).reshape(C, 1)
            dj_dg_T = dj_dg_T + dj_dg_D
            """--new version--"""
            """--old version--
            dj_dg_D = np.zeros(C).reshape(C, 1)
            # inner loop: i to D (I assume D is of length: nr_images)
            for i in range(nr_images):
                weight = 1 / (1 + np.exp(- theta[i, j]))
                term1 = (weight * F[:, j].reshape(C, 1))
                term2 = (S[i, j] * F[:, i]).reshape(C, 1)
                term3 = term1 - term2
                term4 = 2 * GAMMA * (G[:, j].reshape(C, 1) - B[:, j].reshape(C, 1)) + 2 * ETA * G1
                total = term3 + term4
                dj_dg_D = dj_dg_D + total
            dj_dg_T = dj_dg_T + 0.5 * dj_dg_D
               --old version--"""
        # 2.3) update the parameters theta_Y by using back propagation
        new_weights = embedding.backprop_weights(caption_embedder, dj_dg_T)
        caption_embedder.set_weights(new_weights)

    # 3) update B
    B = GAMMA * np.sign(F + G)

    # 4) calculate loss (TODO)
"""
# testing if predictions have changed
new_captions_input = caption_embedder.predict(caption_train_subset.todense())
new_images_input = image_embedder.predict(images_train_subset)

dif = new_captions_input == origional_captions_input
if dif.all():
    print('caption predictions did not change')
else:
    print('caption predictions did change:')
    for i in range(len(dif)):
        if not dif[i].all():
            print('index ' + str(i) + ': ' + str(origional_captions_input[i, :]) + ' -> ' + str(new_captions_input[i, :]))
print('\n\n\n')
dif = new_images_input == origional_images_input
if dif.all():
    print('image predictions did not change')
else:
    print('image predictions did change:')
    for i in range(len(dif)):
        if not dif[i].all():
            print('index ' + str(i) + ': ' + str(origional_images_input[i, :]) + ' -> ' + str(new_images_input[i, :]))
"""
# %%

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
images_input = image_embedder.predict(images_train_subset)

captions = [image_names_c, captions_input]
images = [image_names_i, images_input]
f_score, g_score = ranking.mean_average_precision(captions, images, captions_per_image=captions_per_image)
print('performance on training data: ')
print('f_score = ' + str(round(f_score * 100, 3)) + "%")
print('g_score = ' + str(round(g_score * 100, 3)) + "%")


# testing performance on test data
"""
print('testing performance on testing data')
nr_images = images_test.shape[0]
nr_captions = nr_images * captions_per_image
images_test_subset = images_test.iloc[0:nr_images, 1:].values
caption_test_subset = captions_test_bow[1][0:nr_captions, :]

image_names_c = list()
image_names_i = list()
for i in range(images_test.shape[0]):
    name = captions_train_bow[0][i * captions_per_image][: -2]
    for j in range(captions_per_image):
        image_names_c.append(name)
    image_names_i.append(name)

captions_input = caption_embedder.predict(caption_test_subset.todense())
images_input = image_embedder.predict(images_test_subset)

captions = [image_names_c, captions_input]
images = [image_names_i, images_input]
f_score, g_score = ranking.mean_average_precision(captions, images, captions_per_image=captions_per_image)
print('performance on testing data: ')
print('f_score = ' + str(round(f_score * 100, 3)) + "%")
print('g_score = ' + str(round(g_score * 100, 3)) + "%")
"""


""" everything above this line is part of this class, do not forget to outcomment performance testing on test set!   """


#%%
# --------------------------------------------------------------------
# IMPLEMENTATION BASED ON MATLAB CODE FROM THE PAPER (Vectorized code)
# !! CODE IS NOT WORKING !!
# https://github.com/jiangqy/DCMH-CVPR2017/blob/master/DCMH_matlab/DCMH_matlab/process_DCMH.m
# --------------------------------------------------------------------

# 1) Read in data and preprocess
# ------------------------------
"""
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
"""
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