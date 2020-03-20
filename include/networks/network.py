from numpy import float64
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.layers.merge import Concatenate
from tensorflow_core.python.keras.losses import MeanSquaredError
from tensorflow_core.python.keras.models import Sequential, Model


def custom_distance_loss(y, label):
    """ TODO: implement this loss -> cosine or l2"""
    print('custom loss!')
    return float64(0)


def get_network(input_size):
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
