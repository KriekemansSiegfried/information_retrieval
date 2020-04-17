from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.layers import Dense


def get_image_embedder(input_size, hidden_size=1024, embedding_size=256):
    """ Return a model that embeds an image (feature vector) to an embedding vector"""

    input = Input(shape=(input_size,))
    hidden = Dense(hidden_size, activation='sigmoid')(input)
    output = Dense(embedding_size, activation='tanh')(hidden)
    model = Model(inputs=input, outputs=output)
    print('compiling model')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def get_caption_embedder(input_size, hidden_size=1024, embedding_size=256):
    """ Return a model that embeds a caption (feature vector) to an embedding vector"""
    input = Input(shape=(input_size,))
    hidden = Dense(hidden_size, activation='sigmoid')(input)
    output = Dense(embedding_size, activation='tanh')(hidden)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model
