import sys

from include.bow.dictionary import create_dict, prune_dict
from include.bow.one_hot import convert_to_bow
from include.io.import_captions import import_captions
import numpy as np

# print numpy arrays in full
np.set_printoptions(threshold=sys.maxsize)

caption_filename = '/home/kriekemans/KUL/information_retrieval/dataset/results_20130124.token'

captions = import_captions(caption_filename)

bow_dictionary = create_dict(captions)
bow_dictionary = prune_dict(bow_dictionary)

print('dict -> {}'.format(len(bow_dictionary)))

tokens = [token for token in bow_dictionary.keys()]
print('tokens -> {}'.format(tokens))

vector = convert_to_bow(captions[100], tokens)
print('caption -> {}'.format(captions[100].tokens))
print('bow -> ',vector)
