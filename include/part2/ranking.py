import numpy as np
from scipy.spatial.distance import hamming


def find_lowest(values, nr):
    """
    a helper function that returns the indices of the nr lowest elements
    in the list, only looping through the list once, instead of multiple times, and using
    a large amount of space were the correct indices first found by argsort, the list then
    reorganised, and than taken only the top nr elements of it.
    ! if multiple elements are equal, the order of the inputvalues is preserved !
    :param values:
    :param nr:
    :return:
    """
    indexes = np.argsort(values[0:nr], kind='quicksort')
    current_values = [values[i] for i in indexes]
    current_worst = current_values[0]
    # indices = np.flip(indices)

    for index in range(nr, len(values)):
        value = values[index]
        if value < current_worst:
            j = 1
            while value < current_values[j]:
                j += 1
            if j == 10:
                current_values = current_values[1:9] + [value]
                indexes = indexes[1:9] + [index]
            elif j == 1:
                current_values = [value] + current_values[1:9]
                indexes = [index] + indexes[1:9]
            else:
                current_values = current_values[1:j] + [value] + current_values[j:9]
                indexes = indexes[1:j] + [index] + indexes[j:9]
    return np.flip(indexes)



def find_nearest(input, data_set, nr=10):
    input = input.round()
    data = data_set[1].round()
    distances = np.zeros(len(data), dtype=np.int)
    for i in range(len(data)):
        distances[i] = hamming(input, data[i, :])
    indices = np.argsort(distances, kind='quicksort')
    result_list = [data_set[0][i] for i in np.flip(indices)]
    return result_list[0:nr]


def mean_average_precision(captions, images):
    """
    :param captions: a list of two elements:
        1) captions[0] = a list of image names (strings), corresponding to the caption that resulted in the
                prediction present in the opposite element, that is, if caption[0][n] contains 'xxx.jpg',
                then captions[1][n] contains a hash that was based/predicted on a caption corresponding to
                the image 'xxx.jpg'.
        2) captions[1] = a Nx32 matrix, where N is the number of captions (meaning also len(captions[0])
                should equal N) and 32 is the size of the hashes
    :param images: a list of two elements:
        1) images[0] = a list of image names (strings), corresponding to the image_features that resulted in
                the prediction present in the opposite element, that is, if images[0][n] contains 'xxx.jpg',
                then images[1][n] contains a hash that was based/predicted on the image_feature corresponding
                to the image 'xxx.jpg'.
         2) images[1] = a Mx32 matrix, where M is the number of images (meaning also len(images[0]) should
                equal N) and 32 is the size of the hashes
    :return: a two tuple containing the performance of the performance, respectively for F and G i.e. (f_score, g_score)
    """
    f_score = 0.0
    g_score = 0.0
    nr_captions, nr_images = len(captions[0]), len(images[0])
    caption_matrix = captions[1].round()
    image_matrix = images[1].round()

    # -----------------------------------------------------------------------------
    # perform mAP_10 on captions_predictions
    distances = np.zeros(nr_images, dtype=np.int)
    for j in range(nr_captions):
        image = captions[0][j]
        caption = caption_matrix[j, :]

        # calculate the distances from the prediction to individual images
        for i in range(nr_images):
            distances[i] = hamming(image_matrix[i, :], caption)

        # retrieve the 10 best scoring images
        indexes = find_lowest(distances, 10)
        nearest_10 = [images[0][i] for i in indexes]

        # check if expected result is in nearest_10, and if so, add to the score
        # (this for loop is faster than first checking if the image is present, than finding
        #  its index, since this only iterates through the list at most once, not twice
        for p in range(10):
            if nearest_10[p] == image:
                f_score += (1 / (p + 1))
                break
    f_score = f_score/nr_captions

    # -----------------------------------------------------------------------------
    # perform mAP_10 on image_predictions
    distances = np.zeros(nr_captions, dtype=np.int)
    for j in range(nr_images):
        image = images[0][j]
        feature = image_matrix[j, :]

        # calculate the distances from the prediction to individual images
        for i in range(nr_captions):
            distances[i] = hamming(caption_matrix[i, :], feature)

            # retrieve the 10 best scoring images
        indexes = find_lowest(distances, 10)
        nearest_10 = [captions[0][i] for i in indexes]

        # check if expected result is in nearest_10, and if so, add to the score
        for p in range(10):
            if nearest_10[p] == image:
                g_score += (1 / (p + 1))
                break
    g_score = g_score / nr_images

    return f_score, g_score


