import numpy as np
from numpy.linalg import norm


def rank_images(caption_features, image_features, id_included=False, k=10):
    id_list = None

    if id_included:
        id_list = image_features[:, 0]
        image_features = image_features[:, 1:]

    caption_count = caption_features.shape[0]
    image_count = image_features.shape[0]

    if k is None:
        k = image_count

    caption_norms = []
    image_norms = []
    for index in range(0, caption_count):
        r = caption_features[index, :]
        caption_norms.append(norm(r.astype(np.float)))
        # caption_norms.append(norm(caption_features[index, :]))
    for index in range(0, image_count):
        r = image_features[index, :]
        image_norms.append(norm(r.astype(np.float)))
        # image_norms.append(norm(image_features[index, :]))

    base = [range(0, image_count)]
    empty = [None] * image_count
    result = np.array([[0]*k]*caption_count)
    cos_sim = np.c_[np.transpose(base), np.transpose([empty])]
    for i in range(0, caption_count):
        caption_norm = caption_norms[i]
        for j in range(0, image_count):
            a = caption_features[i, :]
            b = image_features[j, :]
            cos_sim[j, 1] = - np.dot(a.astype(np.float), b.astype(np.float)) / (caption_norm * image_norms[j])

            # cos_sim[j, 1] = - np.dot(caption_features[i, :], image_features[j, :])/(caption_norm*image_norms[j])
        ranked = cos_sim[cos_sim[:, 1].argsort(kind='quicksort')][:k, 0]
        result[i] = ranked

    if not id_included:
        return result

    list_result = []
    for i in range(0, result.shape[0]):
        new_list = []
        for j in range(0, k):
            new_list.append(id_list[result[i, j]])
        list_result.append(np.array(new_list))
    return np.array(list_result)


"""
    for i in range(0, nr):
        cos_sim[i, 1] = - np.dot(caption_feature, image_features[i, :]) / (norm(image_features[i, :]) * caption_norm)

    ranked = cos_sim[cos_sim[:, 1].argsort()]
    return ranked[:, 0]
"""

def run_tests():
    cap_feature1 = np.array([0.9, 0.02, 0.05])
    # cap_feature2 = np.array([0.0, 0.75, 0.95])

    # build an array of caption features
    vect1 = np.array([0.8, 0.02, 0.06])
    vect2 = np.array([0.7, 0.1, 0.1])
    vect3 = np.array([0.95, 0.05, 0.05])
    vect4 = np.array([0.1, 0.8, 0.06])
    vect5 = np.array([0.01, 0.1, 0.9])
    vect6 = np.array([0.6, 0.5, 0.8])
    img_features = np.array([vect1, vect2, vect3, vect4, vect5, vect6])

    # rank the images using the rank_images function
    # check if test result equals expected result
    # print outcome
    res = rank_images(cap_feature1, img_features)
    success = (res == [0, 2, 1, 5, 3, 4])
    if success.all():
        print('test succeeded')
    else:
        print('test failed')

# run_tests()
