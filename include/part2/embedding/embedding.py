from sklearn.preprocessing import Normalizer
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense



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


def advanced_backprop_weights(model, inputs, gradients, learning_rate=0.001):
    weights = model.get_weights()
    for i in range(len(weights)):
        try:
            _ = weights[i].shape[1]
        except IndexError:
            weights[i] = weights[i].reshape((weights[i].shape[0], 1))

    node_values = []
    node_values.append(inputs)
    for layer_weights in weights[:-1]:
        if layer_weights.shape[1] == 1:
            values = node_values[-1] + np.matmul(np.ones((1, layer_weights.shape[0])), layer_weights)
            node_values.append(values)
        else:
            x, y = node_values[-1].shape[0], node_values[-1].shape[1]
            previous_layer_out = np.minimum(node_values[-1], np.ones((x, y)))
            values = np.matmul(np.maximum(previous_layer_out, np.zeros((x, y))), layer_weights)
            node_values.append(values)


    gradients = gradients.reshape((gradients.shape[0], 1))
    # Normalize gradients and retrieve weights
    # gradients = gradients / max(gradients)
    new_weights = []
    weights = weights[::-1]
    node_values = node_values[::-1]
    for layer in range(len(weights)):
        layer_weights = weights[layer]
        values = node_values[layer]
        nodes_in_this_layer = gradients.shape[0]
        nodes_in_next_layer = layer_weights.shape[1]
        if nodes_in_next_layer == 1:
            new_weights.append(layer_weights + learning_rate*gradients)
        else:
            values = values.sum(axis=0).reshape((1, values.shape[1]))
            gradients = np.transpose(np.matmul(gradients, values))
            new_weights.append(layer_weights + learning_rate*gradients)
            gradients = gradients.sum(axis=1).reshape((gradients.shape[0], 1))

    for w in range(len(new_weights)):
        new_w = new_weights[w]
        if new_w.shape[1] == 1:
            new_w = np.squeeze(Normalizer(norm='l2').fit_transform(new_w))
            new_w = new_w.reshape((new_w.shape[0],))
            # elif new_w.shape[0] == 1:  # this case should not happen under normal circumstances
            # print('something wrong')
            # new_w = np.transpose(new_w)
            # new_w = np.squeeze(Normalizer(norm='l2').fit_transform(new_w))
            # new_w = new_w.reshape((new_w.shape[0],))
        else:
            new_w = Normalizer(norm='l2').fit_transform(new_w)
        new_weights[w] = new_w
    # model.set_weights(new_weights[::-1])
    return new_weights[::-1]




