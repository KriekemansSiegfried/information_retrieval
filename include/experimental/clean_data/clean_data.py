import re
import string


# %% load doc into memory
def load_doc(filename, encoding=None):
    """

    :param filename:
    :param encoding:
    :return:
    """
    # open the file as read only
    file = open(filename, 'r', encoding=encoding)
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# extract descriptions for images
def load_descriptions(doc, first_description_only=True, n_desc=None):
    """

    :param doc:
    :param first_description_only:
    :param n_desc:
    :return:
    """
    mapping = dict()
    # process lines
    i = 0
    for line in doc.split('\n'):
        i += 1

        # load only part of the data
        if n_desc is not None and i == n_desc:
            break
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # store the first description for each image only
        if first_description_only:
            if image_id not in mapping:
                mapping[image_id] = image_desc

        # store all descriptions for each image in one big caption
        else:
            if image_id not in mapping:
                mapping[image_id] = image_desc
            else:
                # add image description with a space added
                mapping[image_id] += image_desc.rjust(len(image_id) + 2)

    return mapping


# clean description text
def clean_descriptions(descriptions, min_word_length=3):
    """

    :param descriptions:
    :param min_word_length:
    :return:
    """

    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    for key, desc in descriptions.items():
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each word
        desc = [re_punc.sub('', w) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word) > min_word_length]
        # only store unique words
        unique_desc = []
        for word in desc:
            if word not in unique_desc:
                unique_desc.append(word)
        # store as string
        descriptions[key] = ' '.join(unique_desc)


# save descriptions to file, one per line
def save_doc(descriptions, filename):
    """

    :param descriptions:
    :param filename:
    :return:
    """
    lines = list()
    for key, desc in descriptions.items():
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# load clean descriptions into memory
def load_clean_descriptions(filename):
    """

    :param filename:
    :return:
    """
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # store
        descriptions[image_id] = ' '.join(image_desc)
    return descriptions
