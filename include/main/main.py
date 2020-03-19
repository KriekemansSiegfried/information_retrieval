# %% import libraries

import sys

import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords

# custom defined functions
from include.bow import dictionary
from include.bow import frequent_words as fw
from include.bow import one_hot
from include.io import import_captions
from include.io import import_images
from include.io import output_captions

# to quickly reload functions

# style seaborn for plotting
# %matplotlib qt5 (for interactive plotting)
sns.set()
# print numpy arrays in full
np.set_printoptions(threshold=sys.maxsize)


# %%  import data

# caption_filename = '/home/kriekemans/KUL/information_retrieval/dataset/results_20130124.token'
# image_filename = '/home/kriekemans/KUL/information_retrieval/dataset/image_features.csv'

caption_filename = 'include/data/results_20130124.token'
image_filename = 'include/data/image_features.csv'

# import data
captions = import_captions.import_captions(caption_filename)
images = import_images.import_images(image_filename)


# %% create captions to bow dictionary
bow_dict = dictionary.create_dict(captions)

# %%
# get pandas dataframe with
# most frequent and least frequent words and visualize

# most frequent
df_word_freq = fw.rank_word_freq(dic=bow_dict,
                                 n=20, ascending=False, visualize=True)

# least frequent
df_word_freq = fw.rank_word_freq(dic=bow_dict,
                                 n=20, ascending=True, visualize=True)

# %%
# get stop words
stop_words = set(stopwords.words('english'))

# prune dictionary
bow_dict_pruned, removed_words = dictionary.prune_dict(word_dict=bow_dict,
                                                       stopwords=stop_words, min_word_len=3)

# have a look again at the most frequent words from the updated dictionary
_ = fw.rank_word_freq(dic=bow_dict_pruned, n=20, ascending=False, visualize=True)

# have a look at the removed words
_ = fw.rank_word_freq(dic=removed_words, n=20, ascending=False, visualize=True)

# %% # one hot encode
tokens = list(bow_dict_pruned.keys())
vector = one_hot.convert_to_bow(captions[100], tokens)
print('caption -> {}'.format(captions[100].tokens))
print('bow -> ', vector)

# %%

# write captions to csv + update models/captions with feature
# compress=True means we only store the colnumber of features which are 1
output_captions.output_captions(captions=captions, tokens=tokens,
                                file_name="include/data/caption_features.csv",
                                compress=True)
captions[10].features

# %% import data
# TODO: Make import statement for caption_features.csv
df_image = pd.read_csv("include/data/image_features.csv", sep=" ")

# TODO:
#   1) BOW oke: still fine tuning e.g. example remove words with fewer character than 3
#   2) Image vectors: oke
#   3) make pairs
#   4) Define architecture NN (keras)  + loss functions
