import numpy as np

image_filename = '../data/image_features_triplet.npy'
caption_filename = '../data/caption_features_triplet.npy'


def export_image_features(images, image_model):
    image_features = np.stack(images, axis=0)

    prediction = image_model.predict(image_features)
    np.save(image_filename, prediction)
    print("images saved!")


def export_caption_features(captions, caption_model):
    caption_features = np.stack(map(captions, lambda c: c.features), axis=0)
    prediction = caption_model.predict(caption_features)
    np.save(caption_filename, prediction)


def get_caption_embedding(caption, caption_model):
    input = np.expand_dims(caption.features, axis=0)
    prediction = caption_model.predict(input)
    return np.squeeze(prediction, axis=0)


def get_image_embedding(image, image_model):
    input = np.expand_dims(image.features, axis=0)
    prediction = image_model.predict(input)
    return np.squeeze(prediction, axis=0)
