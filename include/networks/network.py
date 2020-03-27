import tensorflow_core
from numpy import float64
from tensorflow_core.python.keras.backend import transpose, squeeze, map_fn
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.layers.merge import Concatenate
from tensorflow_core.python.keras.losses import MeanSquaredError
from tensorflow_core.python.keras.models import Model, Sequential, load_model
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
        print('custom loss')
        print('y_pred input -> {}'.format(y_pred.shape))
        print('label -> {}'.format(label.shape))

        y_pred = squeeze(transpose(y_pred), axis=1)
        print('y_pred -> {}'.format(y_pred.get_shape()))
        caption_embedding = y_pred[:y_pred.shape[0] // 2]
        image_embedding = y_pred[y_pred.shape[0] // 2:]
        print('caption -> {}'.format(caption_embedding.get_shape()))
        print('image embedding -> {}'.format(image_embedding.get_shape()))

        norm_val = norm(caption_embedding - image_embedding, axis=None)
        print('norm -> {}'.format(norm_val))

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


def get_network_siamese(caption_feature_size, image_feature_size, embedding_size):

    print('input dims = [{}, {}]'.format(caption_feature_size,image_feature_size))
    print('output = {}'.format(embedding_size))

    caption_input = Input(shape=(caption_feature_size,))
    caption_hidden = Dense(4096, activation='relu')(caption_input)
    caption_output = Dense(embedding_size, activation='relu')(caption_hidden)

    image_input = Input(shape=(image_feature_size,))
    image_hidden = Dense(4096, activation='relu')(image_input)
    image_output = Dense(embedding_size, activation='relu')(image_hidden)

    concat = Concatenate()([caption_output, image_output])

    model = Model(inputs=[caption_input, image_input], outputs=concat)
    model.compile(loss=custom_loss(), optimizer='sgd', metrics=['accuracy'])
    return model


def import_network(file_path):
    model = load_model(file_path)
    return model


def export_network(file_path, model):
    model.save(file_path)