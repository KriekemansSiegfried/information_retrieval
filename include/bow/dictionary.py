def preprocess_token(token):
    delete_chars = "\n\t .;,\\"
    return ''.join(c for c in token.lower() if c not in delete_chars)


def create_dict(captions):
    """ Create a dictionary with counts to each word in a given list of sentences """
    word_dict = {}

    for caption in captions:
        for token in caption.tokens:
            clean_token = preprocess_token(token)
            if "\n" in clean_token:
                print('clean token -> {}'.format(clean_token))
            count = word_dict.get(clean_token, 0)
            count += 1
            word_dict[clean_token] = count

    return word_dict


def prune_dict(word_dict, stopwords={}, min_freq=0, max_freq=10 ** 10, min_word_len=0):

    """
    prune words from this dictionary to improve performance of bow representation


    Parameters
    ----------
    word_dict : dictionary
        DESCRIPTION.
    stopwords : set
        user specified set of stopwords that will be deleted
    min_freq: integer
        mininimum frequency the word should occur to be kept
    max_freq: integer
    min_word_len: integer

    Returns
    -------
    word_dict : dictionary
        a pruned dictionary.

    """
    # length of the dictionary before pruning
    dict_removed = dict()
    dict_new = dict()

    # prune dictionary
    for (key, value) in word_dict.items():
        if min_freq < value < max_freq and len(key) >= min_word_len and key not in stopwords:
            dict_new[key] = value
        else:
            dict_removed[key] = value

    print(f'Length dictionary before pruning: {len(word_dict)}')
    print(f'Length dictionary after pruning: {len(dict_new)}')
    print(f'removed {len(dict_removed)} words')

    return dict_new, dict_removed
