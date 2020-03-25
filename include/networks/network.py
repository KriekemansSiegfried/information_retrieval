import tensorflow_core
from tensorflow_core.python.keras.backend import transpose, squeeze
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.layers.merge import Concatenate
from tensorflow_core.python.keras.losses import MeanSquaredError
from tensorflow_core.python.keras.models import Model, Sequential, load_model
from tensorflow_core.python.ops.linalg_ops import norm


def custom_loss():

    def custom_distance_loss(label, y_pred):
        """ TODO: implement this loss -> cosine or l2"""
        print('custom loss')
        y_pred = squeeze(transpose(y_pred))
        print('y_pred -> {}'.format(y_pred.get_shape()))
        caption_embedding = y_pred[:y_pred.shape[0] // 2]
        image_embedding = y_pred[y_pred.shape[0] // 2:]
        print('caption -> {}'.format(caption_embedding.get_shape()))
        print('image embedding -> {}'.format(image_embedding.get_shape()))

        norm_val = norm(caption_embedding - image_embedding, axis=None)
        print('norm -> {}'.format(norm_val))

        return norm_val

    return custom_distance_loss

def get_network(input_size):
    model = Sequential()
    model.add(Dense(input_size, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))

    model.compile('sgd', loss=MeanSquaredError)

    return model


def get_network_siamese(caption_feature_size, image_feature_size, embedding_size):

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


def import_model(file_path):
    model = load_model(file_path)
    return model


def export_model(file_path, model):
    model.save(file_path)