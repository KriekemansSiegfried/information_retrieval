# data pre processing
import numpy as np
from numpy.random import rand, randint, permutation
from sklearn.feature_extraction.text import CountVectorizer
import json

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# own functions
from include.preprocess_data import preprocessing
from include.part2.embedding.embedding import get_caption_embedder, get_image_embedder
from include.part2.embedding import matrix
from include.part2.loss.loss import f_loss, g_loss


# %% GLOBAL VARIABLES (indicated in CAPITAL letters)
PATH = "include/input/"
sns.set()
# %%
"""
Parameters:
"""
gamma = 1
eta = 1
c = 32
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
    path=PATH, document='results_20130124.token', encoding="utf8", dir="include/output/data")

# %% in case you already have ran the cel above once before and don't want to run it over and over
# train
captions_train = json.load(open('include/output/data/train.json', 'r'))
# val
captions_val = json.load(open('include/output/data/val.json', 'r'))
# test
captions_test = json.load(open('include/output/data/test.json', 'r'))

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


#%% prepare for image embedder (train/validation/test)

# 1) image data
# training data has 29783 images but we only select a subset for now
# the validation data and test data has "only" 1000 images each so we don't subset
nr_images_train = 500 # take smaller subset for experimenting
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
captions_pairs_train = captions_train_bow[1][0:captions_per_image*nr_images_train, :]
captions_pairs_val = captions_val_bow[1]
captions_pairs_test = captions_test_bow[1]

print(f"Dimensions of the caption training data: {captions_pairs_train.shape}")
print(f"Dimensions of the caption validation data: {captions_pairs_val.shape}")
print(f"Dimensions of the caption test data: {captions_pairs_test.shape}")


#%%
# -------------------------------------------------
# 2) Make embedding and prepare B, F, G, S matrices
# -------------------------------------------------

# Load image and caption embedder
image_embedder = get_image_embedder(images_pairs_train.shape[1], embedding_size=32)
caption_embedder = get_caption_embedder(captions_pairs_train.shape[1], embedding_size=32)

# Create embedding matrices
# if data points (e.g. caption pairs) are supplied in a compressed format, they will be stored in compressed format
# to convert to a dense format: G.datapoints.todense()
F_train = matrix.EmbeddingMatrix(embedder=image_embedder, datapoints=images_pairs_train)
G_train = matrix.EmbeddingMatrix(embedder=caption_embedder, datapoints=captions_pairs_train)

print('F => {}'.format(F_train))
print('G => {}'.format(G_train))

#%%
# Create theta matrix
theta_train = matrix.ThetaMatrix(F_train, G_train)

# Create similarity (S) matrix of size N*M where N is the number of images and M is the number of captions
image_idx = np.repeat(np.arange(0, nr_images_train, 1), repeats=captions_per_image).tolist()
caption_idx = np.arange(0, nr_images_train*captions_per_image, 1).tolist()
image_caption_pairs = [(image_idx[i], caption_idx[i]) for i in range(len(image_idx))]

S_train = matrix.SimilarityMatrix(image_caption_pairs, nr_images_train, nr_images_train*captions_per_image)

# Create sign matrix (B)
B_train = matrix.SignMatrix(matrix_F=F_train, matrix_G=G_train, gamma=1)

print('S: {}'.format(S_train.matrix.shape))
print('Theta: {}'.format(theta_train.matrix.shape))
print('B: {}'.format(B_train.matrix.shape))
print('F: {}'.format(F_train.matrix.shape))
print('G: {}'.format(G_train.matrix.shape))

# ------------------------------------------------
# 3) Train model on training data
# ------------------------------------------------
# %%

batch_size = 256
epochs = 5
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
        caption_embeddings = caption_embedder.predict(caption_batch.todense())

        # update columns with new embeddings
        np.put(a=F_train.matrix, ind=indices, v=image_embeddings)
        np.put(a=G_train.matrix, ind=indices, v=caption_embeddings)

        theta_train.recalculate()

        # calculate loss values
        loss_f = f_loss(samples=pairs, all_pairs=image_caption_pairs,
                        theta_matrix=theta_train.matrix, F=F_train.matrix,
                        G=G_train.matrix, B=B_train.matrix, S=S_train.matrix)
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

# --------------------------------------------------------------------
# Random Data (depreciated: will be removed in future version)
# --------------------------------------------------------------------

#%%
image_embedder = get_image_embedder(2048, embedding_size=32)
caption_embedder = get_caption_embedder(4096, embedding_size=32)

# dummy values to test with
images = (rand(256, 2048) - 1) * 2  # TODO: fill in real values
captions = (rand(256, 4096) - 1) * 2  # TODO: fill in real values
nr_images = len(images)
nr_captions = len(captions)
image_caption_pairs = [(randint(0, nr_images), randint(0, nr_captions))
                       for i in range(512)]  # TODO: fill in real values

images_pairs = np.array([images[pair[0]] for pair in image_caption_pairs])
captions_pairs = np.array([captions[pair[1]] for pair in image_caption_pairs])
#%%
# Create embedding matrices
F = matrix.EmbeddingMatrix(embedder=image_embedder, datapoints=images_pairs)
G = matrix.EmbeddingMatrix(embedder=caption_embedder, datapoints=captions_pairs)

print('F => {}'.format(F))
print('G => {}'.format(G))
#%%

# Create theta matrix
theta = matrix.ThetaMatrix(F, G)


# Create similarity matrix
S = matrix.SimilarityMatrix(image_caption_pairs, nr_images, nr_captions)

# Create sign matrix
B = matrix.SignMatrix(matrix_F=F, matrix_G=G)

print('S: {}'.format(S.matrix.shape))
print('theta: {}'.format(theta.matrix.shape))
print('B: {}'.format(B.matrix.shape))
print('F: {}'.format(F.matrix.shape))
print('G: {}'.format(G.matrix.shape))
print('B: {}'.format(B.matrix))
# %%
# take samples
batch_size = 32
epochs = 1
all_indices = np.arange(len(image_caption_pairs))

f_loss_sums = []
g_loss_sums = []
all_indices = permutation(all_indices)
for j in range(epochs):
    # permute batches

    for index, i in enumerate(range(0, len(image_caption_pairs), batch_size)):
        batch_interval = (i, min(len(image_caption_pairs), i + batch_size))
        # print('batch: {}'.format(batch_interval))

        indices = all_indices[batch_interval[0]:batch_interval[1]]
        pairs = [image_caption_pairs[index] for index in indices]
        # print('indices -> {}'.format(indices))
        image_batch = images[[image_caption_pairs[index][0] for index in indices]]
        caption_batch = captions[[image_caption_pairs[index][1] for index in indices]]

        # make predictions for the batch
        image_embeddings = image_embedder.predict(image_batch)
        caption_embeddings = caption_embedder.predict(caption_batch)

        # replace columns with new embeddings
        np.put(a=F.matrix, ind=indices, v=image_embeddings)
        np.put(a=G.matrix, ind=indices, v=caption_embeddings)

        theta.recalculate()

        # calculate loss values
        loss_f = f_loss(samples=pairs, all_pairs=image_caption_pairs,
                        theta_matrix=theta.matrix, F=F.matrix,
                        G=G.matrix, B=B.matrix, S=S.matrix)
        loss_g = g_loss(samples=pairs, all_pairs=image_caption_pairs,
                        theta_matrix=theta.matrix, F=F.matrix,
                        G=G.matrix, B=B.matrix, S=S.matrix)

        print('f loss ({}) -> {}'.format(loss_f.shape, loss_f[0]))
        print('g loss ({}) -> {}'.format(loss_g.shape, loss_g[0]))

        f_loss_sums.append(np.sum(loss_f[0]))
        g_loss_sums.append(np.sum(loss_g[0]))

        # update weights based on these loss values
        F.update_weights(loss_f)
        G.update_weights(loss_g)

        # recalculate sign matrix
        B.recalculate()
        print("batch {} done".format(index))
# %%
print(B)
print('samples generated')

print('f losses -> {}'.format(f_loss_sums))
plt.plot(f_loss_sums)
plt.ylabel('sums of f_loss values')
plt.show()

print('g losses -> {}'.format(g_loss_sums))
plt.plot(g_loss_sums)
plt.ylabel('sums of g_loss values')
plt.show()
