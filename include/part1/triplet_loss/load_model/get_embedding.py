import numpy as np


def get_caption_embedding(caption, caption_model, reshape=True):
    """

    :param caption:
    :param caption_model:
    :param reshape:
    :return:
    """

    if reshape:
        caption = np.expand_dims(caption, axis=0)
    prediction = caption_model.predict(caption)
    if reshape:
        return np.squeeze(prediction, axis=0)
    else:
        return prediction


def get_image_embedding(image, image_model, reshape=True):
    """

    :param image:
    :param image_model:
    :param reshape:
    :return:
    """

    if reshape:
        image = np.expand_dims(image, axis=0)
    prediction = image_model.predict(image)
    if reshape:
        return np.squeeze(prediction, axis=0)
    else:
        return prediction
