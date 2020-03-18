import numpy as np
from include.bow.dictionary import preprocess_token


def get_one_hot(token, tokens):

    """

    :param token:
    :param tokens:
    :return:
    """
    clean_token = preprocess_token(token)
    if clean_token not in tokens:
        # Cannot return one-hot representation of token that is not in list of given tokens
        return None
    else:
        vector = np.zeros(len(tokens), dtype=int)
        index = tokens.index(clean_token)
        vector[index] = 1

        return vector


def convert_to_bow(caption, token_set):

    """
    :param caption:
    :param token_set:
    :return:
    """

    tokens = caption.tokens
    vector = np.zeros(len(token_set))

    for token in tokens:
        token_bow = get_one_hot(token, token_set)
        if token_bow is None:
            # unknown word, skip
            continue
        vector = np.add(vector, token_bow)

    return vector
