# -------------------------------------------------------------------------------------------------------------
#  %% 0) Load libraries
# -------------------------------------------------------------------------------------------------------------

# data manipulations
import json
from math import ceil
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse.linalg import svds
from numpy.random import permutation
from sklearn.feature_extraction.text import CountVectorizer

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch
import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam

# own functions
from include.preprocess_data import preprocessing
from include.preprocess_data.word2vec import convert_to_word2vec
# from include.part2.loss.loss import f_loss_torch
# from include.part2.torch_embedding.caption_embedder import CaptionEmbedder
from include.part2.torch_embedding.embedder import Embedder
from include.part2 import ranking
from include.part2.embedding import embedding
from include.part2.embedding import matrix

# %%
# -------------------------------------------------------------------------------------------------------------
#  %% 1) GLOBAL VARIABLES (indicated in CAPITAL letters)
# -------------------------------------------------------------------------------------------------------------

BASE = "include/"  # check your own path
PATH = BASE + "input/"

# Parameters (given in the assignment)
GAMMA = 1
ETA = 1
C = 32

# plotting
sns.set()

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
#        - DONE: perform map10 on testing data
#        - DONE: perform map10 on testing data


#%%

def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss


# %%
# -------------------------------------------------------------------------------------------------------------
# 2) Read in data and preprocess
# -------------------------------------------------------------------------------------------------------------

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
captions_train = preprocessing.clean_descriptions(
     descriptions=captions_train, min_word_length=2, stem=stemming, unique_only=False
)
captions_val = preprocessing.clean_descriptions(
     descriptions=captions_val, min_word_length=2, stem=stemming, unique_only=False
)
captions_test = preprocessing.clean_descriptions(
     descriptions=captions_test, min_word_length=2, stem=stemming, unique_only=False
)

# %% convert to bow vectors

# c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
# fit on training output (descriptions)
# c_vec.fit(captions_train.values())
# print(f"Size vocabulary: {len(c_vec.vocabulary_)}")

# # transform on train/val/test output
# captions_train_vec = [list(captions_train.keys()), c_vec.transform(captions_train.values())]
# captions_val_vec = [list(captions_val.keys()), c_vec.transform(captions_val.values())]
# captions_test_vec = [list(captions_test.keys()), c_vec.transform(captions_test.values())]

# %% convert to word2 vectors

caption_train_keys = captions_train.keys()
caption_test_keys = captions_test.keys()
captions_val_keys = captions_val.keys()

captions_train, captions_test, captions_val = convert_to_word2vec((captions_train, captions_val, captions_test))

captions_train_vec = [list(caption_train_keys), captions_train]
captions_val_vec = [list(captions_val_keys), captions_val]
captions_test_vec = [list(caption_test_keys), captions_test]

# %%
# -------------------------------------------------------------------------------------------------------------
# 3) Initialize matrices: F, G, B, S
# -------------------------------------------------------------------------------------------------------------


nr_images = 500  # only on a subset of the data
captions_per_image = 5
nr_captions = nr_images * captions_per_image

# Define torch variable containing similarity information of captions and images
block = np.ones(captions_per_image**2).reshape(captions_per_image, captions_per_image)
S = Variable(torch.from_numpy((np.kron(np.eye(nr_images, dtype=int), block))))


# F and G: shape (C, D) where D is the number of image_caption pairs
F_buffer = torch.randn(nr_captions, C)
G_buffer = torch.randn(nr_captions, C)

# initialize B: shape: (C,D)
B = torch.sign(F_buffer + G_buffer)

# take a smaller part of the data = nr_images
image_train_subset = np.repeat(a=images_train.iloc[0:nr_images, 1:].values, repeats=5, axis=0)
caption_train_subset = captions_train_vec[1][0:nr_captions, :]


# %%
# -------------------------------------------------------------------------------------------------------------
# 4) Load networks and train
# -------------------------------------------------------------------------------------------------------------
image_embedder = Embedder(image_train_subset.shape[1], C)
caption_embedder = Embedder(caption_train_subset.shape[1], C)

# TODO: optimize lr value
image_optimizer = SGD(image_embedder.parameters(), lr=0.005)
caption_optimizer = SGD(caption_embedder.parameters(), lr=0.005)

batch_size = 64
epochs = 25
data_size = nr_images * captions_per_image

ones_batch = torch.ones(batch_size)
ones_other = torch.ones(data_size - batch_size)

x_loss_values = []
y_loss_values = []
loss_values = []

# %%  Training loop

for epoch in range(epochs):
    print('Starting epoch {}'.format(epoch + 1))
    # image update loop
    for i in range(data_size // batch_size):
        # get random set of indices as batch
        batch_indices = np.random.permutation(data_size)[:batch_size]
        other_indices = np.setdiff1d(range(data_size), batch_indices)

        batch_images = Variable(torch.from_numpy(image_train_subset[batch_indices, :]))

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

        theta_x = 1.0 / 2 * torch.matmul(batch_image_embedding, G.t())
        logloss_x = -torch.sum(sim * theta_x - torch.log(1.0 + torch.exp(theta_x)))
        quantization_x = torch.sum(torch.pow(B_var[batch_indices, :] - batch_image_embedding, 2))
        balance_x = torch.sum(torch.pow(
            torch.matmul(batch_image_embedding.t(), ones_batch) + torch.matmul(F[other_indices].t(),
                                                                               ones_other), 2))
        loss_x = logloss_x + GAMMA * quantization_x + ETA * balance_x
        loss_x /= (data_size)

        x_loss_values.append(loss_x.detach().numpy().item())

        # print('x loss {}'.format(loss_x.detach().numpy().item()))

        image_embedder.zero_grad()
        loss_x.backward()
        image_optimizer.step()

        parameters = image_embedder.parameters()
        params = []
        for param in parameters:
            params.append(param)
        # print('iter done')

    for i in range(data_size // batch_size):
        # get random set of indices as batch
        batch_indices = np.random.permutation(data_size)[:batch_size]
        other_indices = np.setdiff1d(range(data_size), batch_indices)

        batch_captions = Variable(torch.from_numpy(caption_train_subset[batch_indices, :]))

        # TODO: is this correct?
        sim = S[batch_indices, :]

        batch_text_embedding = caption_embedder(batch_captions)
        G_buffer[batch_indices, :] = batch_text_embedding
        F = Variable(F_buffer)
        G = Variable(G_buffer)
        B_var = Variable(B)
        ones_batch_var = Variable(ones_batch)
        ones_other_var = Variable(ones_other)

        theta_y = 1.0 / 2 * torch.matmul(batch_text_embedding, F.t())
        logloss_y = -torch.sum(sim * theta_y - torch.log(1.0 + torch.exp(theta_y)))
        quantization_y = torch.sum(torch.pow(B_var[batch_indices, :] - batch_text_embedding, 2))
        balance_y = torch.sum(torch.pow(
            torch.matmul(batch_text_embedding.t(), ones_batch_var) + torch.matmul(G[other_indices].t(), ones_other_var),
            2))
        loss_y = logloss_y + GAMMA * quantization_y + ETA * balance_y
        loss_y /= (data_size)

        y_loss_values.append(loss_y.detach().numpy().item())

        # print('y loss {}'.format(loss_y.detach().numpy().item()))

        caption_optimizer.zero_grad()
        loss_y.backward()
        caption_optimizer.step()

    # update sign matrix
    B = torch.sign(F_buffer + G_buffer)

    loss_epoch = calc_loss(B, F_buffer, G_buffer, S, GAMMA, ETA)
    loss_values.append(loss_epoch.data)
    print('epoch loss: {}'.format(loss_epoch))

# %%
# -------------------------------------------------------------------------------------------------------------
# 5 ) Evaluate training
# -------------------------------------------------------------------------------------------------------------

# santiy check
print(f"Number of positive bits: {sum(sum(B > 0))}")
print(f"Number of negative bits: {sum(sum(B < 0))}")

# testing performance on training data
print('testing performance on training data')

x_val_labels = [i for i in range(0, len(x_loss_values))]
y_val_labels = [i for i in range(0, len(y_loss_values))]

sns.set()
sns.lineplot(x=x_val_labels, y=x_loss_values)
plt.show()
sns.lineplot(x=y_val_labels, y=y_loss_values)
plt.show()
sns.lineplot(x=np.arange(0, epochs), y=np.array(loss_values))
plt.show()

#%%
# -------------------------------------------------------------------------------------------------------------
# 6 ) MAP@10 Performance
# -------------------------------------------------------------------------------------------------------------
"""
A) IMPLEMENTATION 1
"""

# 1) IMAGES: make predictions
# only make a prediction for every 5th, image vectors don't change, only captions do

images = Variable(torch.from_numpy(image_train_subset[::5]))
image_embedding = np.sign(image_embedder(images).data.detach().numpy())

# 2) CAPTIONS: make predictions
captions = Variable(torch.from_numpy(caption_train_subset))
caption_embedding = np.sign(caption_embedder(captions).data.detach().numpy())


# %% 2) computing ranking images
from include.ranking import ranking as ranking1

image_id = images_train.iloc[0:nr_images, 0].values
caption_id = np.array(captions_train_vec[0][0:nr_captions])

ranking_images = ranking1.rank_embedding(
    caption_embed=caption_embedding,
    caption_id=caption_id,
    image_embed=image_embedding,
    image_id=image_id,
    retrieve="images",
    k=10,
    distance_metric="Hamming",
    add_correct_id=True
)

# %% 2 b) compute ranking captions
ranking_captions = ranking1.rank_embedding(
    caption_embed=caption_embedding,
    caption_id=caption_id,
    image_embed=image_embedding,
    image_id=image_id,
    retrieve="captions",
    k=10,
    distance_metric="Hamming",
    add_correct_id=True
)
# %%
for key, value in ranking_captions.items():
    list_ranking = [item.split(".")[0] for item in value[0]]

# %% 3 a) compute MAP@10 images
average_precision_images = ranking1.average_precision(ranking_images, gtp=1)
print(f"{average_precision_images.head()}")
print(f"Mean average precision @10 is: {round(average_precision_images.mean()[0]*100, 4)} %")

# %% 3 b) compute MAP@10 captions
average_precision_captions = ranking1.average_precision(ranking_captions, gtp=5)
print(f"{average_precision_captions.head()}")
print(f"Mean average precision @10 is: {round(average_precision_captions.mean()[0]*100, 4)} %")



# %%
"""
B) IMPLEMENTATION 2
"""

from include.part2 import ranking as ranking2

image_names_c = list()
image_names_i = list()
for i in range(nr_images):
    name = captions_train_vec[0][i * captions_per_image][: -2]
    for j in range(captions_per_image):
        image_names_c.append(name)
    image_names_i.append(name)

captions_input = caption_embedder(torch.from_numpy(caption_train_subset))
images_input = image_embedder(torch.from_numpy(image_train_subset))

captions_input = captions_input.detach().numpy()
images_input = images_input.detach().numpy()

captions = [image_names_c, captions_input]
images = [image_names_i, images_input]
f_score, g_score = ranking2.mean_average_precision(captions, images, captions_per_image=captions_per_image)
print('performance on training data: ')
print('f_score = ' + str(f_score * 100) + "%")
print('g_score = ' + str(g_score * 100) + "%")
