import gensim
from nltk import word_tokenize
import numpy as np
import nltk
nltk.download('punkt')


def convert_to_word2vec(captions):
    data = []
    for caption_set in captions:
        for caption_id in caption_set.keys():
            caption_array = [word for word in word_tokenize(caption_set[caption_id])]
            data.append(caption_array)
    model = gensim.models.Word2Vec(data, min_count=1,
                                   size=200, window=5)

    dicts = []
    for caption_set in captions:
        caption_vector_dict = []
        for caption_id in caption_set.keys():
            vector = np.zeros(model.vector_size)
            tokens = word_tokenize(caption_set[caption_id])
            for word in tokens:
                vector += model.wv[word]
            vector /= len(tokens)
            caption_vector_dict.append((vector))
        caption_vector_dict = np.stack(caption_vector_dict, axis=0)
        dicts.append(caption_vector_dict)
    return dicts
