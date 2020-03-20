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
            lo = max(0, pos - neg + margin)         #<- should this not be 'neg - pos' if we want to minimize loss?
            print(lo)
            loss = loss + lo
            #loss = loss + max(0, pos - neg + margin)
    return loss


def caption_img_loss(caption_feature, pos_img_features, neg_img_features, margin=0):
    return img_caption_loss(caption_feature, pos_img_features, neg_img_features, margin=margin)



# Loss function test code

#build img_feature
img_feature1 = np.array([0.9, 0.02, 0.05])
img_feature2 = np.array([0.0, 0.75, 0.95])

#build pos_caption_feature
vect1 = np.array([0.8, 0.02, 0.06])
vect2 = np.array([0.7, 0.1, 0.05])
vect3 = np.array([0.95, 0.2, 0.1])
pos_caption_features = np.array([vect1, vect2, vect3])
#pos_caption_features.add(vect1)
#pos_caption_features.add(vect2)
#pos_caption_features.add(vect3)

#build neg_caption_feature
vect4 = np.array([0.1, 0.8, 0.06])
vect5 = np.array([0.01, 0.1, 0.9])
vect6 = np.array([0.6, 0.5, 0.8])
neg_caption_features = np.array([vect4, vect5, vect6])

print(img_caption_loss(img_feature1, pos_caption_features, neg_caption_features))
print('-----')
print(img_caption_loss(img_feature2, pos_caption_features, neg_caption_features))