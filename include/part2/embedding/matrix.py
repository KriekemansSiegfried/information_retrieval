import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow_core.python.keras import backend
from scipy.sparse import csr_matrix


class Matrix:

    def __init__(self):
        self.matrix = None

    def __repr__(self):
        return str(self.matrix)


class EmbeddingMatrix(Matrix):
    """
    Represents a matrix that contains the embeddings of a series of datapoints
    """

    def __init__(self, embedder, datapoints):
        super().__init__()
        self.embedder = embedder
        self.embedding_size = embedder.layers[-1].output.shape[1]
        self.datapoints = datapoints
        # check if datapoint are in a sparse matrix (e.g. caption data)
        if isinstance(datapoints, csr_matrix):
            # covert to dense matrix first to predict
            self.matrix = np.transpose(self.embedder.predict(self.datapoints.todense()))
        else:
            self.matrix = np.transpose(self.embedder.predict(self.datapoints))
        self.learning_rate = 0.001

    def _sigmoid_derivative(self, value):
        return value
        # return value * (1 - value)

    def update_weights(self, loss_value):
        """
        This function implements backpropagation based on the given loss value
        """
        errors = []
        weights = self.embedder.get_weights()
        new_weights = []
        # loop of layers in reverse order
        for layer_weights in weights[::-1]:
            if not errors:
                # if last layer, get loss value and update based on that
                # TODO: Should a derivative be calculated here?
                loss_value = loss_value * -1
                errors.append(loss_value)
                new_w = layer_weights + loss_value * self.learning_rate
                new_w = np.squeeze(Normalizer(norm='l2').fit_transform(np.expand_dims(new_w, axis=0)))
                new_weights.append(new_w)
            else:
                # new weights are based on the error of the next layer,
                # since this loop is reverse, it is the previous error
                last_error = errors[-1]
                if len(layer_weights.shape) == 2:
                    # If layer_weights is 2-dimensional, need to apply weighted sum for each weight update
                    new_error = np.array([np.dot(last_error, row) for row in layer_weights])
                    # print('new error -> {}'.format(new_error))
                    new_w = layer_weights + np.matmul(new_error, layer_weights) * self.learning_rate
                    new_w = Normalizer(norm='l2').fit_transform(new_w)
                    new_weights.append(new_w)
                    errors.append(new_error)
                else:
                    # Intermediary rows that are not the last layer do
                    # not contain any values (I suppose not used by keras?)
                    new_weights.append(layer_weights)
        errors
        # layer weights should be reversed so input layer is first, output layer is last
        new_weights = new_weights[::-1]
        self.embedder.set_weights(new_weights)
        return new_weights


class SignMatrix(Matrix):
    """
    Represents a matrix consisting of 1/0's
    A point is equal to ysign(F + G)
    """

    def __init__(self, matrix_F: EmbeddingMatrix, matrix_G: EmbeddingMatrix, gamma=1):
        assert (matrix_F.matrix.shape == matrix_G.matrix.shape)
        super().__init__()
        self.gamma = gamma
        self.matrix_F = matrix_F
        self.matrix_G = matrix_G
        self.matrix = np.empty(shape=matrix_F.matrix.shape)
        self.recalculate()

    def recalculate(self):
        self.matrix = self.gamma * np.sign(self.matrix_F.matrix + self.matrix_G.matrix)
        self.matrix[self.matrix == 0] = 1
        return self.matrix

    def update(self, samples):
        pass


class ThetaMatrix(Matrix):
    """
    This represents 1/2 * F^T * G
    """

    def __init__(self, F, G):
        super().__init__()
        self.F = F
        self.G = G
        self.matrix = None
        self.recalculate()

    def recalculate(self):
        self.matrix = 1 / 2 * np.matmul(np.transpose(self.F.matrix), self.G.matrix)


class SimilarityMatrix(Matrix):
    """
    This matrix represents the similarity of images and captions
    [i,j] == 1 if image at index i and caption at index j should be similar (are a pair)
    """

    def __init__(self, pairs, nr_images, nr_captions):
        super().__init__()
        self.matrix = np.zeros(shape=(nr_images, nr_captions), dtype=np.int8)
        for (index_image, index_caption) in pairs:
            self.matrix[index_image, index_caption] = 1  # value is 1 if pair exists AKA image and caption are similar