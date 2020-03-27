import tensorflow_core
from numpy import float64
import tensorflow_core.python.keras.backend as K
from tensorflow_core.python.keras.backend import transpose, squeeze, map_fn
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.layers.core import Lambda
from tensorflow_core.python.keras.layers.merge import Concatenate
from tensorflow_core.python.keras.losses import MeanSquaredError
from tensorflow_core.python.keras.models import Model, Sequential, load_model
from tensorflow_core.python.keras.optimizers import Adam
from tensorflow_core.python.ops.linalg_ops import norm


def custom_loss():
    def row_distance(y_pred):
        """ function to map a tensor to l2 distances
        This function assumes that both label and y are related to 1 row of data

        """
        print('y_pred -> {}'.format(y_pred.shape))
        # print('label -> {}'.format(label.shape))

        return float64(10)

    def custom_distance_loss(label, y_pred):
        """ TODO: implement this loss -> cosine or l2"""
        print('custom loss called')

        y_pred = squeeze(transpose(y_pred), axis=1)
        caption_embedding = y_pred[:y_pred.shape[0] // 2]
        image_embedding = y_pred[y_pred.shape[0] // 2:]

        norm_val = norm(caption_embedding - image_embedding, axis=None)

        return norm_val

    return custom_distance_loss


def get_network(input_size, layers, output_size, input_dim=None, output_activation='relu',
                loss=MeanSquaredError, optimizer=None, metrics=None):
    """

    :param input_size:  bit_size of the input layer?
    :param layers: an array of integer values, entries being bit_size for every intermediate layer
    :param output_size: size of the output layer
    :param input_dim: dimension of the input
    :param output_activation: output activation function, default set to 'relu'
    :param loss: the loss-function
    :param optimizer:
    :param metrics:
    :return:
    """
    model = Sequential()
    if input_dim is None:
        model.add(Dense(input_size, activation='relu'))
    else:
        model.add(Dense(32, activation='relu', input_dim=input_dim))
    for layer in layers:
        model.add(Dense(layer, activation='relu'))
    model.add(Dense(output_size, activation=output_activation))
    if optimizer is None:
        if metrics is None:
            model.compile(loss=loss)
        else:
            model.compile(loss=loss, metrics=metrics)
    else:
        if metrics is None:
            model.compile(loss=loss, optimizer=optimizer)
        else:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    return model


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def get_network_siamese(caption_feature_size, image_feature_size, embedding_size):
    print('input dims = [{}, {}]'.format(caption_feature_size, image_feature_size))
    print('output = {}'.format(embedding_size))

    caption_input = Input(shape=(caption_feature_size,))
    caption_hidden = Dense(4096, activation='relu')(caption_input)
    caption_output = Dense(embedding_size, activation='relu')(caption_hidden)

    image_input = Input(shape=(image_feature_size,))
    image_hidden = Dense(4096, activation='relu')(image_input)
    image_output = Dense(embedding_size, activation='relu')(image_hidden)

    concat = Concatenate()([caption_output, image_output])

    # dist = Lambda(euclidean_distance, output_shape=lambda x: x[0])(caption_output, image_output)

    model = Model(inputs=[caption_input, image_input], outputs=concat)
    model.compile(loss=custom_loss(), optimizer='sgd', metrics=['accuracy'])
    return model


def get_network_siamese_contrastive(caption_size, image_feature_size, embedding_size):
    caption_input = Input(shape=(caption_size,))
    caption_hidden = Dense(4096, activation='relu')(caption_input)
    caption_output = Dense(embedding_size, activation='relu')(caption_hidden)

    image_input = Input(shape=(image_feature_size,))
    image_hidden = Dense(4096, activation='relu')(image_input)
    image_output = Dense(embedding_size, activation='relu')(image_hidden)

    l2_layer = Lambda(lambda x: K.sqrt(x[0] - x[1]))
    l2_distance = l2_layer([caption_output, image_output])

    prediction = Dense(1, activation='sigmoid')(l2_distance)

    model = Model(inputs=[caption_input, image_input], outputs=prediction)
    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')

    return model


def triplet_loss(y_true, y_pred, alpha=0.4):
    length = y_pred.shape.as_list()[-1]

    negative = y_pred[:, 0:int(length / 3)]
    positive = y_pred[:, int(length / 3):int(length * 2 / 3)]
    anchor = y_pred[:, int(length * 2 / 3):int(length)]

    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = pos_dist - neg_dist + alpha
    regularized_loss = K.maximum(loss, 0.0)

    return regularized_loss

def get_network_triplet_loss(caption_size, image_size, embedding_size):
    def base_model(size):
        input = Input(shape=(size,))
        hidden = Dense(4096, activation='relu')(input)
        output = Dense(embedding_size, activation='relu')(hidden)

        return [input, output]

    caption_input_neg, caption_output_neg = base_model(caption_size)
    caption_input_pos, caption_output_pos = base_model(caption_size)
    image_input, image_output = base_model(image_size)

    concat = Concatenate()([caption_output_neg, caption_output_pos, image_output])
    model = Model([caption_input_neg, caption_input_pos, image_input], concat)
    model.compile(loss=triplet_loss, optimizer='adam')
    model.summary()
    return model

    def import_network(file_path):
        model = load_model(file_path)
        return model

    def export_network(file_path, model):
        model.save(file_path)
