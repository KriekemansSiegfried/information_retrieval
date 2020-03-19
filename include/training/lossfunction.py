import numpy as np
from numpy.linalg import norm


def img_caption_loss(img_feature, pos_caption_features, neg_caption_features, margin=0):
    loss = 0
    img_norm = norm(img_feature)
    positives = set()
    negatives = set()

    for positive_feature in pos_caption_features:
        cos_sim = np.dot(positive_feature, img_feature) / (norm(positive_feature) * img_norm)
        positives.add(cos_sim)

    for negative_feature in neg_caption_features:
        cos_sim = np.dot(negative_feature, img_feature) / (norm(negative_feature) * img_norm)
        negatives.add(cos_sim)

    for pos in positives:
        for neg in negatives:
            loss = loss + max(0, pos - neg + margin)
    return loss


def caption_img_loss(caption_feature, pos_img_features, neg_img_features, margin=0):
    return img_caption_loss(caption_feature, pos_img_features, neg_img_features, margin=margin)
