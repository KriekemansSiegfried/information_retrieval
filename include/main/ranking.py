import numpy as np
from numpy.linalg import norm


def rank_images(caption_feature, image_features, id_included=False):
    nr = image_features.shape[0]
    print(nr)
    if id_included:
        base = image_features[:, 0]
        image_features = image_features[:, 1:]
    else:
        base = np.transpose([range(0, nr)])

    empty = [None]*nr
    cos_sim = np.append(base, np.transpose([empty]), 1)
    print(cos_sim)
    caption_norm = norm(caption_feature)
    for i in range(0, nr):
        cos_sim[i, 1] = - np.dot(caption_feature, image_features[i, :]) / (norm(image_features[i, :]) * caption_norm)

    ranked = cos_sim[cos_sim[:, 1].argsort()]
    return ranked[:, 0]


#testing code
cap_feature1 = np.array([0.9, 0.02, 0.05])
cap_feature2 = np.array([0.0, 0.75, 0.95])

# build pos_caption_feature
vect1 = np.array([0.8, 0.02, 0.06])
vect2 = np.array([0.7, 0.1, 0.1])
vect3 = np.array([0.95, 0.05, 0.05])
vect4 = np.array([0.1, 0.8, 0.06])
vect5 = np.array([0.01, 0.1, 0.9])
vect6 = np.array([0.6, 0.5, 0.8])
img_features = np.array([vect1, vect2, vect3, vect4, vect5, vect6])

res = rank_images(cap_feature1, img_features)
success = (res == [0, 2, 1, 5, 3, 4])
if success.all():
    print('test succeeded')
else:
    print('test failed')
