import numpy as np


def convert_to_dataset(pairs):
    """ convert all pairs to dataset with labels 0/1: negative/positive"""

    labels = []
    caption_features_set = []
    image_features_set = []

    for (key, positive, negative) in pairs:
        image_features = key.features
        positive_caption_features = positive.features
        negative_caption_features = negative.features

        # add positive example
        image_features_set.append(image_features)
        caption_features_set.append(positive_caption_features)
        labels.append(np.float64(1))

        # add negative example
        image_features_set.append(image_features)
        caption_features_set.append(negative_caption_features)
        labels.append(np.float64(0))

    image_features_set = np.stack(image_features_set, axis=0)
    caption_features_set = np.stack(caption_features_set, axis=0)

    print('caption_feature_set -> {}'.format(caption_features_set.shape))

    labels = np.stack(labels, axis=0)

    return [image_features_set, caption_features_set], labels


def convert_to_triplet_dataset(triplets):
    """ convert all pairs to dataset with labels 0/1: negative/positive"""

    labels = []
    caption_features_set_neg = []
    caption_features_set_pos = []
    image_features_set = []

    for (key, positive, negative) in triplets:
        image_features_set.append(key.features)
        caption_features_set_pos.append(positive.features)
        caption_features_set_neg.append(negative.features)
        labels.append(0)  # dummy label

        # yield [negative.features, positive.features, key.features], np.float64(1)

    image_features_set = np.stack(image_features_set, axis=0)
    caption_features_set_pos = np.stack(caption_features_set_pos, axis=0)
    caption_features_set_neg = np.stack(caption_features_set_neg, axis=0)
    labels = np.stack(labels, axis=0)
    return [caption_features_set_neg, caption_features_set_pos, image_features_set], labels
