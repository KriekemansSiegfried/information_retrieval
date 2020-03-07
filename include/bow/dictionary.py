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


def prune_dict(word_dict):
    """ prune words from this dictionary to improve performance of bow representation """
    # TODO: extend this dict
    stop_words = ['a','in', 'with', 'to', 'from', '']

    for word in stop_words:
        if word in word_dict:
            del word_dict[word]

    return word_dict
