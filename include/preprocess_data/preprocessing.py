import json
import os
import re
import string
import numpy as np
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
from include.util.util import print_progress_bar


def get_split_idx(path=None):
    """
    Reads in the train, validation and test indices

    :param path: String
        directory of the train, validation and test indices, and image feature vectors
    :return: List
        with the train, validation and test indices
    """

    # read in train train/validation/test indices
    train_idx = pd.read_csv(path + 'train_idx.txt', dtype=str, header=None).values.flatten()
    val_idx = pd.read_csv(path + 'val_idx.txt', dtype=str, header=None).values.flatten()
    test_idx = pd.read_csv(path + 'test_idx.txt', dtype=str, header=None).values.flatten()

    return train_idx, val_idx, test_idx


def read_split_images(path=None, verbose=True):
    """
    reads in the image model and input into a train, validation and test set
    according to https://github.com/BryanPlummer/flickr30k_entities

    :param path: String
        directory of the train, validation and test indices, and image feature vectors
    :param verbose: Boolean
        print out the dimensions of the train, validation and test set
    :return: Pandas Data frame
        with the images feature vectors split in a train, validation and test set

    """

    # read in train train/validation/test indices
    train_idx, val_idx, test_idx = get_split_idx(path=path)

    # load first the whole image dataset in
    df = pd.read_csv(path + "image_features.csv", sep=" ", header=None)

    # remove .jpg from first row
    df.iloc[:, 0] = df.iloc[:, 0].str.split('.').str[0]

    # split image model in train/validation/test set
    df_train = df[df.iloc[:, 0].isin(train_idx.tolist())]
    df_val = df[df.iloc[:, 0].isin(val_idx.tolist())]
    df_test = df[df.iloc[:, 0].isin(test_idx.tolist())]

    if verbose:
        print(f"Dimensions of the training set: {df_train.shape}")
        print(f"Dimensions of the validation set: {df_val.shape}")
        print(f"Dimensions of the test set: {df_test.shape}")

    return df_train, df_val, df_test


# %% load doc into memory
def load_doc(filename=None, encoding=None):
    """

    :param filename: String
        name of your file
    :param encoding: String
        encoding is the name of the encoding used to decode or encode the
        file. This should only be used in text mode. The default encoding is
        platform dependent, but any encoding supported by Python can be
        passed.  See the codecs module for the list of supported encodings.
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
def read_split_captions(path=None, document=None, encoding=None, verbose=True, dir=None):
    """

    :param path: String
        path where to find the data
    :param document: String
        name of your file to import
    :param encoding: String
        encoding is the name of the encoding used to decode or encode the
        file. This should only be used in text mode. The default encoding is
        platform dependent, but any encoding supported by Python can be
        passed.  See the codecs module for the list of supported encodings.
    :param verbose: Boolean
        print out the length of the train, validation and test set
    :return: Dictionary

    """

    # read in raw document
    doc = load_doc(path + document, encoding=encoding)

    # store caption in train validation and test set
    train = dict()
    val = dict()
    test = dict()
    # get train/val/test indices to split dataset
    train_idx, val_idx, test_idx = get_split_idx(path)
    # for progress bar
    n = len(train_idx) + len(val_idx) + len(test_idx)
    i = 0
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # ignore lines shorter than two
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)

        if image_id.split(".")[0] in train_idx:
            train[image_id] = image_desc
        elif image_id.split(".")[0] in val_idx:
            val[image_id] = image_desc
        elif image_id.split(".")[0] in test_idx:
            test[image_id] = image_desc
        else:
            print(f'{image_id} not in train/validation or test set')

        # update progress
        i += 1
        print_progress_bar(i=i, maximum=n * 5, post_text="Finish", n_bar=20)

    if verbose:
        print("\n")
        print(f"Dimensions of the training set: {len(train)}")
        print(f"Dimensions of the validation set: {len(val)}")
        print(f"Dimensions of the test set: {len(test)}")

    if dir is not None:
        json.dump(train, open(os.path.join(dir, "train.json"), 'w'))
        json.dump(val, open(os.path.join(dir, "val.json"), 'w'))
        json.dump(test, open(os.path.join(dir, "test.json"), 'w'))
        if verbose:
            print(f"saved train/validation and test set in directory: {dir}")
    return train, val, test


# clean description text
def clean_descriptions(descriptions, min_word_length=3, stem=True, unique_only=False, verbose=True):
    """

    :param descriptions: Dictionary
        containing the captions
    :param min_word_length: Integer
        indicating the minimum length of a word to be kept
    :param stem: Boolean
        Apply stemming, stemming based on the Lancaster (Paice/Husk) stemming algorithm (default True)
    :param unique_only: Boolean
        Keep for for each caption only the unique words (default False)
    :param verbose: Boolean
        print progress and other information (Default True)
    :return: Dictionary
        cleaned captions
    """
    # stemmer
    if stem:
        if verbose:
            print("Stemming based on the Lancaster (Paice/Husk) stemming algorithm")
        st = LancasterStemmer()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # for progress bar
    if verbose:
        i = 0
        n = len(descriptions)
    for key, desc in descriptions.items():
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each word
        desc = [re_punc.sub('', w) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word) > min_word_length]
        # stem words
        if stem:
            desc = [st.stem(word) for word in desc]
        # only store unique words
        if unique_only:
            unique_desc = []
            for word in desc:
                if word not in unique_desc:
                    unique_desc.append(word)
            # store as string
            descriptions[key] = ' '.join(unique_desc)
        else:
            descriptions[key] = ' '.join(desc)

        # update progress
        if verbose:
            i += 1
            print_progress_bar(i=i, maximum=n, post_text="Finish", n_bar=20)
    print("\n")
    return descriptions


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


def convert_to_triplet_dataset(captions=None, images=None, captions_k=5, p=100,
                               n_row=None, todense=False, verbose=True):
    """

    :param captions: Numpy matrix in Compressed Sparse Row format
        Caption matrix
    :param images: Numpy array
        Image matrix (already one hot encoded)
    :param captions_k: Integer
         Number of captions per image (default 5)
    :param p: Integer
        The number of images used to compute the similarity.
        The images with the lowest mean squared error are uses to select the caption from
    :param n_row: Integer
        Number of rows to iterate over (default uses all images)
    :param todense: Boolean
        Whether to convert the Numpy sparse matrix to a dense format
    :param verbose: Boolean
        Print progress and other information (Default True)
    :return:
        1) caption of the negative and positive examples and
        2) caption feature set of the negative, positive examples and image feature set
        3) labels full of zero (needed for keras)
    """

    n_images = images.shape[0]
    n_captions = n_images * captions_k
    if n_row is not None:
        n_images = min(n_row, n_images)
        n_captions = n_images * captions_k

    # keep track of the positive image id's of the positive examples
    caption_ids_pos = captions[0][0:n_captions]
    # get the captions features
    caption_features_set_pos = captions[1][0:n_captions, :]
    # our image model set has 5 times less examples than our caption model, so we need to repeat
    image_features_set = np.repeat(images.values[0:n_images, 1:], repeats=captions_k, axis=0).astype(dtype=float)
    # indices for the negative examples
    negatives_idx = []
    # just add labels of zero (Keras does need this)
    labels = np.zeros(n_captions)

    # make a random sample of triplet_sample
    # when computing the difference, we make the assumption that one of the 5 captions is sufficient
    for i in range(0, n_captions, captions_k):
        # get image_id: format is '1000092795', because you don't
        # want captions of a different # but the from same image as a negative
        caption_id_pos = caption_ids_pos[i].split(".")[0]

        sample = True
        while sample:
            # make a random draw to get p negative samples
            idx_sample = np.random.choice(a=n_captions, size=p, replace=False)
            # get sampled negative image_ids
            sampled_neg_caption_ids = [caption_ids_pos[j].split(".")[0] for j in idx_sample]
            # check if image_id is not in sampled idx
            if caption_id_pos not in sampled_neg_caption_ids:
                sample = False

        # get the positive image example
        image_pos_example = image_features_set[i]
        # get the negative image examples
        image_neg_matrix = image_features_set[idx_sample]

        # compute the distance
        dist = [np.power((image_pos_example - image_neg_matrix[idx]), 2).sum() for idx in range(p)]
        # find the index (idx) of the closest rep_image distance
        min_distance_idx = np.argpartition(dist, kth=range(captions_k))[0:captions_k].tolist()
        # add index of the negative example to the negative index list
        negatives_idx += idx_sample[min_distance_idx].tolist()
        # update progress
        if verbose:
            print_progress_bar(i=i, maximum=n_captions, post_text="Finish", n_bar=20)
    print("\n")

    # get negative  caption id and negative captions features
    caption_ids_neg = np.array(caption_ids_pos)[negatives_idx]
    caption_features_set_neg = caption_features_set_pos[negatives_idx]

    # convert sparse matrix to full matrix
    if todense:
        caption_features_set_neg = caption_features_set_neg.todense()
        caption_features_set_pos = caption_features_set_pos.todense()

    return (caption_ids_neg, caption_ids_pos), \
           [caption_features_set_neg, caption_features_set_pos, image_features_set], labels


def load_captions(load_from_json=False,
                  path_raw_data="include/input/results_20130124.token",
                  encoding="utf8",
                  dir_to_read_save=None):

    # check if we can read from json format
    if load_from_json and dir_to_read_save is not None:
        try:
            captions = json.load(open(os.path.join(dir_to_read_save, "all_data.json"), 'r'))
        except ValueError:
            print(f"File 'all_data.json' not found in {dir_to_read_save}")

    else:
        # open the file as read only
        file = open(path_raw_data, 'r', encoding=encoding)
        # read all text
        text = file.read()
        # close the file
        file.close()

        captions = dict()

        # process lines
        for line in text.split('\n'):
            # split line by white space
            tokens = line.split()
            # ignore lines shorter than two
            if len(line) < 2:
                continue
            # take the first token as the image id, the rest as the description
            caption_id, caption_desc = tokens[0], tokens[1:]
            # convert description tokens back to string
            caption_desc = ' '.join(caption_desc)
            captions[caption_id] = caption_desc
        # save
        if dir_to_read_save is not None:
            if not os.path.exists(dir_to_read_save):
                os.makedirs(dir_to_read_save)
            json.dump(captions, open(os.path.join(dir_to_read_save, "all_data.json"), 'w'))

    return captions
