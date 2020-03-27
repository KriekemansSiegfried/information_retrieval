# %% import libraries

import sys

import numpy as np
import scipy.sparse as sparse
import seaborn as sns
from nltk.corpus import stopwords

# custom defined functions
from tensorflow_core.python.keras.callbacks import EarlyStopping

from include.bow import dictionary, one_hot
from include.io import import_captions, import_images, output_captions
# style seaborn for plotting
# %matplotlib qt5 (for interactive plotting: run in the python console)
from include.networks.network import get_network_siamese, get_network_siamese_contrastive, get_network_triplet_loss
# to quickly reload functions
from include.training.dataset import convert_to_dataset, convert_to_triplet_dataset
from include.util.pairs import get_pairs_images, make_dict

sns.set()
# print numpy arrays in full
np.set_printoptions(threshold=sys.maxsize)

# %%  import data


# caption_filename = '/home/kriekemans/KUL/information_retrieval/dataset/results_20130124.token'
# image_filename = '/home/kriekemans/KUL/information_retrieval/dataset/image_features.csv'

caption_filename = '../data/results_20130124.token'
image_filename = '../data/image_features.csv'
# read in data
captions = import_captions.import_captions(caption_filename)
images = import_images.import_images(image_filename)

print('loaded {} captions'.format(len(captions)))
print('loaded {} images'.format(len(images)))

# %% create captions to bow dictionary
bow_dict = dictionary.create_dict(captions)

# %%
# get pandas dataframe with
# most frequent and least frequent words and visualize

# %%
# get stop words
stop_words = set(stopwords.words('english'))

# prune dictionary
bow_dict_pruned, removed_words = dictionary.prune_dict(word_dict=bow_dict,
                                                       stopwords=stop_words,
                                                       min_word_len=3,
                                                       min_freq=10,
                                                       max_freq=1000)

# have a look again at the most frequent words from the updated dictionary
# _ = fw.rank_word_freq(dic=bow_dict_pruned, n=20, ascending=False, visualize=True)

# have a look at the removed words
# _ = fw.rank_word_freq(dic=removed_words, n=20, ascending=False, visualize=True)

# %% # one hot encode
tokens = list(bow_dict_pruned.keys())

print('converting caption features')

caption_feature_size = len(tokens)

progress = 0
pruned_captions = []
for caption in captions:
    progress += 1
    if progress % 2500 == 0:
        print(progress)
    # efficiently store sparse matrix
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    caption.features = one_hot.convert_to_bow(caption, tokens)
    pruned_captions.append(caption)
    if (progress == 5000):
        break
# %%

captions = pruned_captions

print('features converted')
#
print('creating triplet dict')
pair_dict = make_dict(images, captions)
print('creating triplets')
pairs = get_pairs_images(pair_dict)
print('pairs created')
print('creating dataset with labels')
dataset, labels = convert_to_triplet_dataset(pairs)
print('dataset created')
print('network loading')
network = get_network_triplet_loss(caption_feature_size,len(images[0].features), 256)
print('network loaded')
network.fit(dataset, labels, epochs=10, use_multiprocessing=True, callbacks=[EarlyStopping(monitor='loss', patience=20)])
print('network fitted')

# print('features converted')
#
# print('creating pair dict')
# pair_dict = make_dict(images, captions)
# print('creating pairs')
# pairs = get_pairs_images(pair_dict)
#
#
# print('pairs created')
# print('creating dataset with labels')
# dataset, labels = convert_to_dataset(pairs)
# print('dataset created')
# print('network loading')
# network = get_network_siamese(len(images[0].features), caption_feature_size, 256)
# print('network loaded')
#
# print('input_2 -> {}'.format(dataset[1].shape))
# network.fit(dataset, labels, batch_size=1)
# print('network fitted')


# %% output captions to compressed format + update captions.features
# output_captions.output_captions(captions=captions, tokens=tokens,
#                                 file_name="include/data/caption_features.npz",
#                                 n_rows=len(captions))
# # representation
# print(captions[10].features)

# # %% load caption features in compressed format
#
# df_captions = sparse.load_npz('include/data/caption_features.npz')
# # if you want to go to the uncompressed format
# # df_captions_uncomp = df_captions.todense()
#
# # %%images (normal format) (this is in pandas dataframe format) (31782, 2049)
# df_image = pd.read_csv("include/data/image_features.csv", sep=" ", header=None)
#
# # TODO:
# #   1) Define train/valdiation/test (and train_valdation ==> for training your final network)
# #   2) Define architecture NN (keras)
# #   3) loss functions
