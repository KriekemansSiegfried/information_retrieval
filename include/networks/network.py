from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import square
from tensorflow.python.keras.layers import BatchNormalization, Dense, Lambda, concatenate
from tensorflow.python.keras.losses import MeanSquaredError
# def cosine_distance(jd, jt):
#     jd = K.l2_normalize(jd, axis=-1)
#     jt = K.l2_normalize(jt, axis=-1)
#     return -K.mean(jd * jt, axis=-1, keepdims=True)
#
# def get_network(bow_length, feature_vector_size):
#         ip1 = Input(bow_length,)
#         ip2 = Input(feature_vector_size,)
#
#         backbone = Sequential()
#         backbone.add(Dense(50, activation='relu'))
#         backbone.add(Dense(200, activation='sigmoid'))
#
#         op1,op2 = backbone(ip1), backbone(ip2)
#
#         model = Model(inputs=[ip1,ip2], outputs=[op1,op2])
#
#         model.compile('sgd', loss=cosine_distance)
#
#         return model
from tensorflow.python.ops.gen_control_flow_ops import Merge


def get_network(input_size):
    model = Sequential()
    model.add(Dense(input_size, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))

    model.compile('sgd', loss=MeanSquaredError)

    return model


def get_network_siamese(caption_feature_size, image_feature_size, embedding_size):
    caption_network = Sequential()

    caption_network.add(Dense(caption_feature_size, activation='relu'))
    caption_network.add(Dense(4096, activation='relu'))
    caption_network.add(Dense(2048, activation='relu'))
    caption_network.add(BatchNormalization())

    image_network = Sequential()

    image_network = Sequential()

    image_network.add(Dense(image_feature_size, activation='relu'))
    image_network.add(Dense(4096, activation='relu'))
    image_network.add(Dense(2048, activation='relu'))
    image_network.add(BatchNormalization())

    caption_output = caption_network.layers[len(caption_network.layers) - 1]
    image_output = image_network.layers[len(image_network.layers) - 1]

    merge = Lambda(lambda x: square(x[1].output - x[0].output), output_shape=lambda x: x[0])


    merge_loss = concatenate([caption_network, image_network], axis=1)

    model = Sequential()
    model.add(merge_loss)

    model.compile(loss='sgd', optimizer='adam', metrics=['accuracy'])

    return model
