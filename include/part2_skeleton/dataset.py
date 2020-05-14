import json
from random import random, uniform, randrange, sample, randint

import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import torch as pt

from part2_skeleton.preprocessing import read_split_images
from part2_skeleton.word2vec import train_w2v, use_w2v

PATH = "input/"


class FLICKR30K(Dataset):
    w2v_model = None
    count_model = None

    def __init__(self, mode="train", limit=-1, word_transformer="w2v"):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        internal_set_images = read_split_images(path=PATH, mode=self.mode, limit=limit)
        internal_set_captions = json.load(open('output/data/{}.json'.format(self.mode), 'r'))

        self.image_labels = internal_set_images.iloc[:, 0]
        internal_set_images = internal_set_images.drop(0, 1)
        self.images = internal_set_images.to_numpy()
        self.caption_labels = list(internal_set_captions.keys())

        if word_transformer == "w2v":
            if mode == 'train':
                self.captions, FLICKR30K.w2v_model = train_w2v(internal_set_captions.values())
            else:
                self.captions = use_w2v(internal_set_captions.values(), FLICKR30K.w2v_model)
        elif word_transformer == "bow":
            if mode == 'train':
                FLICKR30K.c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
                self.captions = FLICKR30K.c_vec.fit_transform(internal_set_captions.values())
            else:
                # transform on train/val/test set
                self.captions = FLICKR30K.c_vec.transform(internal_set_captions.values())
        else:
            print("word_transformer argument should be either w2v or bow")

        if limit > -1:
            self.captions = self.captions[:limit * 5]
            self.caption_labels = self.caption_labels[:limit * 5]

        self.captions_per_image = len(self.caption_labels) / len(self.image_labels)
        self.images = np.repeat(self.images, repeats=self.captions_per_image,
                                axis=0)

        self.image_indices = np.random.permutation(len(self.images))
        self.caption_indices = self.create_caption_indices(self.image_indices)

        self.captions = self.captions[self.caption_indices]
        self.caption_labels = self.captions[self.caption_indices]

        self.images = self.images[self.image_indices]
        self.image_labels = self.images[self.image_indices]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.image_indices[index], pt.tensor(self.images[index]).float(), self.caption_indices[
                index], pt.tensor(
                self.captions[index]).float()
        else:
            ## return in order: always linked to eachother
            return pt.tensor(self.images[self.image_indices[index]]).float(), pt.tensor(
                self.captions[self.caption_indices[index]]).float()

    def __len__(self):
        return len(self.captions)

    def get_dimensions(self):
        return self.images.shape[1], self.captions.shape[1]

    def create_caption_indices(self, image_indices):
        permutation = np.random.permutation(len(self.images))
        offsets = [randint(0,4) for _ in range(len(self.images))]
        for index in range(len(self.images)):
            if uniform(0, 1) <= .5:
                base_index = self.image_indices[index] - self.image_indices[index] % 5
                permutation[index] = base_index + offsets[index]
        return permutation
