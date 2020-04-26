from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import Dense
from sklearn.preprocessing import Normalizer
import numpy as np


def get_image_embedder(input_size, hidden_size=1024, embedding_size=256):
    """ Return a model that embeds an image (feature vector) to an embedding vector"""
    input = Input(shape=(input_size,))
    hidden = Dense(hidden_size, activation='sigmoid')(input)
    output = Dense(embedding_size, activation='tanh')(hidden)
    model = Model(inputs=input, outputs=output)
    print('compiling model')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model


def get_caption_embedder(input_size, hidden_size=1024, embedding_size=256):
    """ Return a model that embeds a caption (feature vector) to an embedding vector"""
    input = Input(shape=(input_size,))
    hidden = Dense(hidden_size, activation='sigmoid')(input)
    output = Dense(embedding_size, activation='tanh')(hidden)
    model = Model(inputs=input, outputs=output)
    print('compiling model')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model


def store_embeder(model, name):
    model.save(name)


def simplefied_backprop_weights(model, gradients, learning_rate=0.001):
    # if shape is (32,1), transform to (32,). If shape already is (32,), do nothing
    try:
        gradients = gradients[:, 0]
    except IndexError:
        pass
    # normalize gradients and retrieve weights
    gradients = gradients/max(gradients)
    weights = model.get_weights()
    new_weights = []
    # iterate through the weights in reverse order
    for layer_weights in weights[::-1]:
        # check the dimension of the weights
        # if dimension is of form (??,), it will be reset to (??,1)
        try:
            dimension = layer_weights.shape[1]
        except IndexError:
            dimension = 1
            layer_weights = layer_weights.reshape(layer_weights.shape[0], 1)
        if dimension == 1:
            suplement = np.transpose(gradients*np.ones((1, 1), dtype=np.float))
        else:
            suplement = gradients*np.ones((layer_weights.shape[0], 1), dtype=np.float)
        gradients = suplement.sum(axis=1)
        gradients = gradients/max(gradients)
        layer_weights = layer_weights + suplement*learning_rate
        if dimension == 1:
            layer_weights = layer_weights.reshape(layer_weights.shape[0],)
        new_weights.append(layer_weights)
    new_weights = new_weights[::-1]
    model.set_weights(new_weights)


def backprop_weights(model, gradients, learning_rate=0.001):
    # If shape is (32,1), transform to (32,).
    # If shape already is (32,), do nothing.
    try:
        gradients = gradients[:, 0]
    except IndexError:
        pass
    # Normalize gradients and retrieve weights
    gradients = gradients/max(gradients)
    weights = model.get_weights()
    new_weights = []
    for layer_weights in weights[::-1]:
        # If weights.shape is (??,), no new gradients need be
        # calculated for next layer, otherwise, try will succeed, and
        # new gradients are calculated for the next layer
        x = gradients.shape[0]
        signs = np.sign(gradients)
        for i in range(x):
            if signs[i] == 0:
                signs[i] = 0.5
        try:
            y = layer_weights.shape[1]
            new_layer_weights = np.zeros((layer_weights.shape[0], y), dtype=np.float32)
            for i in range(layer_weights.shape[1]):
                addition = signs*np.maximum(np.absolute(gradients), 0.001*np.ones((x, 1))[:, 0])  # <-
                new_layer_weights[i, :] = layer_weights[i, :] + addition * learning_rate        # <-
                # new_layer_weights[i, :] = layer_weights[i, :] + gradients * learning_rate
            # gradients = np.array([np.dot(gradients, row) for row in new_layer_weights])
            gradients = np.array([np.dot(gradients, row) for row in layer_weights])
            gradients = gradients / max(0.01, max(gradients))
        except IndexError:
            addition = signs * np.maximum(np.absolute(gradients), 0.1 * np.ones((x, 1))[:, 0])  # <-
            new_layer_weights = layer_weights + addition * learning_rate                        # <-
            # new_layer_weights = layer_weights + gradients * learning_rate
        new_weights.append(new_layer_weights)
    # model.set_weights(new_weights[::-1])
    return new_weights[::-1]





