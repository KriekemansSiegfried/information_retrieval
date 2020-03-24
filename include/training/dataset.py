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
        labels.append(1)

        # add negative example
        image_features_set.append(image_features)
        caption_features_set.append(negative_caption_features)
        labels.append(0)

    image_features_set = np.stack(image_features_set, axis=0)
    caption_features_set = np.stack(caption_features_set, axis=0)
    labels = np.stack(labels, axis=0)

    print('images -> {}'.format(image_features_set.shape))
    print('caption_features_set -> {}'.format(caption_features_set.shape))
    print('labels -> {}'.format(labels.shape))

    print('{} pairs '.format(len(pairs)))
    return [image_features_set, caption_features_set], labels

