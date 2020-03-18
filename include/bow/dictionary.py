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


def prune_dict(word_dict, stopwords={}, min_freq=0, max_freq=0):
    
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

    Returns
    -------
    word_dict : dictionary
        a pruned dictionary.

    """
    # length of the dictionary before pruning
    n_old = len(word_dict)
    
    # remove stopwords from dictionary
    if stopwords !={}:
        for word in stopwords:
            if word in word_dict:
                del word_dict[word]

    
    # filter dictionary based on minimum  and maximum frequency
    if min_freq >0 | max_freq > 0:
        word_dict = { key:value for (key,value) in word_dict.items() if value > min_freq or value < max_freq}
    
    # length of the dictionary after pruning
    n_new = len(word_dict)
    
    print(f'Length dictionary before pruning: {n_old}')
    print(f'Length dictionary after pruning: {n_new}')
    print(f'removed {n_old-n_new} words')

    return word_dict
