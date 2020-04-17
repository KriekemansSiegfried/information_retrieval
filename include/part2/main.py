# data preprocessing
import numpy as np
from numpy.random import rand, randint, permutation
from sklearn.feature_extraction.text import CountVectorizer
import json

# visualization
import matplotlib.pyplot as plt

# own functions
from include.preprocess_data import preprocessing
from include.part2.embedding.embedding import get_caption_embedder, get_image_embedder
from include.part2.embedding.matrix import EmbeddingMatrix, SignMatrix, SimilarityMatrix, ThetaMatrix
from include.part2.loss.loss import f_loss, g_loss


# %% GLOBAL VARIABLES (indicated in CAPITAL letters)
PATH = "include/input/"

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

# %% read in image output
image_train, image_val, image_test = preprocessing.read_split_images(path=PATH)

# %% read in caption output and split in train, validation and test set and save it
caption_train, caption_val, caption_test = preprocessing.read_split_captions(
    path=PATH, document='results_20130124.token', encoding="utf8", dir="include/output/data")

# %% in case you already have ran the cel above once before and don't want to run it over and over
# train
caption_train = json.load(open('include/output/data/train.json', 'r'))
# val
caption_val = json.load(open('include/output/data/val.json', 'r'))
# test
caption_test = json.load(open('include/output/data/test.json', 'r'))

# %% clean captions (don't run this more than once or
# you will prune your caption dictionary even further as it has the same variable name)

# experiment with it: my experience: seems to work better if you apply stemming when training
stemming = True
caption_train = preprocessing.clean_descriptions(
    descriptions=caption_train, min_word_length=2, stem=stemming, unique_only=False
)
caption_val = preprocessing.clean_descriptions(
    descriptions=caption_val, min_word_length=2, stem=stemming, unique_only=False
)
caption_test = preprocessing.clean_descriptions(
    descriptions=caption_test, min_word_length=2, stem=stemming, unique_only=False
)

# %% convert to bow
c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
# fit on training output (descriptions)

c_vec.fit(caption_train.values())
print(f"Size vocabulary: {len(c_vec.vocabulary_)}")
# transform on train/val/test output
caption_train_bow = [list(caption_train.keys()), c_vec.transform(caption_train.values())]
caption_val_bow = [list(caption_val.keys()), c_vec.transform(caption_val.values())]
caption_test_bow = [list(caption_test.keys()), c_vec.transform(caption_test.values())]

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

# Create embedding matrices
F = EmbeddingMatrix(embedder=image_embedder, datapoints=images_pairs)
G = EmbeddingMatrix(embedder=caption_embedder, datapoints=captions_pairs)

print('F => {}'.format(F))
print('G => {}'.format(G))

# Create theta matrix
theta = ThetaMatrix(F, G)

# get data

# Create similarity matrix
S = SimilarityMatrix(image_caption_pairs, nr_images, nr_captions)

# Create sign matrix
B = SignMatrix(matrix_f=F, matrix_g=G)

print('S: {}'.format(S.matrix.shape))
print('theta: {}'.format(theta.matrix.shape))
print('B: {}'.format(B.matrix.shape))
print('F: {}'.format(F.matrix.shape))
print('G: {}'.format(G.matrix.shape))

print('B: {}'.format(B.matrix))

# take samples
batch_size = 32
epochs = 15
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
        image_batch = images[[image_caption_pairs[i][0] for index in indices]]
        caption_batch = captions[[image_caption_pairs[i][0] for index in indices]]

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
