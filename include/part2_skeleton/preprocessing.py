import json
import os
import re
import string
import sys

import numpy as np
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer


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


def read_split_images(path=None, mode='train', limit=-1):
    """
    reads in the image output and input into a train, validation and test set
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

    # split image output in train/validation/test set
    if mode == 'train':
        df_train = df[df.iloc[:, 0].isin(train_idx.tolist())]
        if limit > -1:
            df_train = df_train.iloc[:limit]
        print(f"Dimensions of the training set: {df_train.shape}")
        return df_train
    elif mode == 'val':
        df_val = df[df.iloc[:, 0].isin(val_idx.tolist())]
        if limit > -1:
            df_val = df_val.iloc[:limit]
        print(f"Dimensions of the validation set: {df_val.shape}")
        return df_val
    elif mode == 'test':
        df_test = df[df.iloc[:, 0].isin(test_idx.tolist())]
        if limit > -1:
            df_test = df_test.iloc[:limit]
        print(f"Dimensions of the test set: {df_test.shape}")
        return df_test
    else:
        print('Invalid mode: should be in (train,test,val)')
        sys.exit(0)


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


