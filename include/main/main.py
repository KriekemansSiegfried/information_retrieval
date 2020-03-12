# imort libraries
import sys
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords


# custom defined functions
from include.bow.dictionary import create_dict, prune_dict
from include.bow.one_hot import convert_to_bow
from include.io.import_captions import import_captions
from include.bow.frequent_words import rank_word_freq

# style seaborn for plotting
sns.set()

# print numpy arrays in full
np.set_printoptions(threshold=sys.maxsize)

# caption_filename = '/home/kriekemans/KUL/information_retrieval/dataset/results_20130124.token'
caption_filename = 'include/data/results_20130124.token'

captions = import_captions(caption_filename)

bow_dictionary = create_dict(captions)

# get pandas dataframe with most frequent and least frequent words and visualize

# most frequent
df_word_freq = rank_word_freq(dic=bow_dictionary, n=20, 
                              ascending=False, visualize=True)
# least frequent
df_word_freq = rank_word_freq(dic=bow_dictionary, n=20, 
                              ascending=True, visualize=True)

# stop words
stop_words = set(stopwords.words('english'))

# play with the min_freq (it drops rather quickly)
bow_dictionary = prune_dict(bow_dictionary, stopwords=stop_words, min_freq=0)

# have a look again at the most frequent words
_ = rank_word_freq(dic=bow_dictionary, n=20, 
                              ascending=False, visualize=True)

# remove space 
bow_dictionary = prune_dict(bow_dictionary, stopwords=set(""), min_freq=0)



tokens = [token for token in bow_dictionary.keys()]
print('tokens -> {}'.format(tokens))


vector = convert_to_bow(captions[100], tokens)
print('caption -> {}'.format(captions[100].tokens))
print('bow -> ',vector)



# TODO: 1) BOW oke: still fine tuning e.g. example remove words with fewer character than 3 
#  2) Image vectors: oke
# 3) make pairs
# 4) Neural networkds: (see keras) loss functions









