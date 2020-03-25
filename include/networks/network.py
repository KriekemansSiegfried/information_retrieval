import numpy as np
from numpy.linalg import norm
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.layers.merge import Concatenate
from tensorflow_core.python.keras.losses import MeanSquaredError
from tensorflow_core.python.keras.models import Model, Sequential, load_model


def custom_distance_loss(label, y_pred):
    """ TODO: implement this loss -> cosine or l2"""
    print('custom loss! -> {} , {}'.format(label, y_pred.shape))
    print('split point -> {}'.format(y_pred.shape[1] / 2))

    embeddings = np.split(y_pred.numpy(), 2, axis=1)
    caption_embedding = embeddings[0]
    image_embedding = embeddings[1]

    print('embeddings -> {}'.format(embeddings))
    return norm(caption_embedding - image_embedding)


def get_network(input_size):
    """

    :param input_size:
    :return:
    """

    model = Sequential()
    model.add(Dense(input_size, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))

    model.compile('sgd', loss=MeanSquaredError)

    return model


def get_network_siamese(caption_feature_size, image_feature_size, embedding_size):
    print('caption -> {}'.format(caption_feature_size))
    print('image -> {}'.format(image_feature_size))
    print('embedding -> {}'.format(embedding_size))

    caption_input = Input(shape=(caption_feature_size,))
    caption_hidden = Dense(4096, activation='relu')(caption_input)
    caption_output = Dense(embedding_size, activation='relu')(caption_hidden)

    image_input = Input(shape=(image_feature_size,))
    image_hidden = Dense(4096, activation='relu')(image_input)
    image_output = Dense(embedding_size, activation='relu')(image_hidden)

    concat = Concatenate()([caption_output, image_output])

    model = Model(inputs=[caption_input, image_input], outputs=concat)
    model.compile(loss=custom_distance_loss, optimizer='adam', metrics=['accuracy'])
    return model


def import_network(file_path):
    model = load_model(file_path)
    return model


def export_network(file_path, model):
    model.save(file_path)