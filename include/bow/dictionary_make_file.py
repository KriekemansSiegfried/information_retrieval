from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import scipy.sparse as sparse
import numpy as np
from include.bow import dictionary, frequent_words as fw, one_hot
from include.io import import_captions

from sklearn.feature_extraction.text import CountVectorizer

caption_filename = 'include/data/results_20130124.token'
visualize = False

# import captions from caption file (defined by filename above)
# create a dictionary (bow_dict), which is a set of the form
# {<word> : <#appearance_of_word> , <word> : <#appearance_of_word> , ...}
captions = import_captions.import_captions(caption_filename)
bow_dict = dictionary.create_dict(captions)
print('loaded {} captions'.format(len(captions)))

# if visualization turned on:
#   display the dictionary from least frequent to most frequent
#   display the dictionary from most frequent to least frequent
if visualize:
    df_word_freq = fw.rank_word_freq(dic=bow_dict, n=20, ascending=False, visualize=True)
    df_word_freq = fw.rank_word_freq(dic=bow_dict, n=20, ascending=True, visualize=True)
    df_word_freq = None

# get/import stop words
# prune the dictionary
stop_words = set(stopwords.words('english'))
a, b = dictionary.prune_dict(word_dict=bow_dict, stopwords=stop_words, min_word_len=3, min_freq=0, max_freq=1000)
bow_dict_pruned, removed_words = a, b

tokens = bow_dict_pruned.keys()
print(tokens)

vectorizer = CountVectorizer(vocabulary=set(tokens))
print(vectorizer.vocabulary)



# testing
test_string = "Two young guys with shaggy hair look at their hands while hanging out in the yard"
vector = vectorizer.transform([test_string])
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())




# tokenize and build vocab
# summarize

"""
tokenizer = Tokenizer(num_words=len(tokens)+1)
tokenizer.fit_on_texts(tokens)
num_words = len(tokenizer.word_index) + 1
print(num_words)
print(tokenizer)


test_string = 'Two young guys with shaggy hair look at their hands while hanging out in the yard'
caption = captions[0]
caption.features = sparse.csr_matrix(one_hot.convert_to_bow(caption, tokens), dtype=np.int8)

sequence = tokenizer.texts_to_sequences(test_string)
print(sequence)
"""






