# imort libraries
import sys
import numpy as np
import seaborn as sns


# custom defined functions
from include.bow.dictionary import create_dict, prune_dict
from include.bow.one_hot import convert_to_bow
from include.io.import_captions import import_captions
from include.bow.frequent_words import rank_word_freq

sns.set()

# print numpy arrays in full
np.set_printoptions(threshold=sys.maxsize)

# caption_filename = '/home/kriekemans/KUL/information_retrieval/dataset/results_20130124.token'
caption_filename = 'include/data/results_20130124.token'

captions = import_captions(caption_filename)

bow_dictionary = create_dict(captions)

# get pandas dataframe with most frequent words and visualize
df_word_freq = rank_word_freq(dic=bow_dictionary, n=20, 
                              ascending=False, visualize=True)

# stop words
# take most 10 frequent stopwords and manually update list
# TODO: update list
stop_words  = set(df_word_freq['word'].iloc[0:10])
stop_words = stop_words.union({'two','are','to','at','an'})

# play with the min_freq (it drops rather quickly)
bow_dictionary = prune_dict(bow_dictionary, stopwords=stop_words, min_freq=5)

# have a look again at the most frequent words
_ = rank_word_freq(dic=bow_dictionary, n=20, 
                              ascending=False, visualize=True)

tokens = [token for token in bow_dictionary.keys()]
print('tokens -> {}'.format(tokens))


vector = convert_to_bow(captions[100], tokens)
print('caption -> {}'.format(captions[100].tokens))
print('bow -> ',vector)
