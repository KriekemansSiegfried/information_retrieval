import numpy as np
from tensorflow_core.python.keras import backend


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
        self.matrix = np.transpose(self.embedder.predict(self.datapoints))
        self.learning_rate = 0.000001

    def update_weights(self, loss_value):
        """
        This function implements backpropagation based on the given loss value
        """
        errors = []
        weights = self.embedder.get_weights()
        new_weights = []
        for layer_weights in weights[::-1]:
            if not errors:
                # if last layer, get loss value and update based on that
                # TODO: Should a derivative be calculated here?
                errors.append(loss_value)
                new_w = layer_weights + loss_value * self.learning_rate
                new_weights.append(new_w)
            else:
                # new weights are based on the error of the next layer,
                # since this loop is reverse, it is the previous error
                last_error = errors[-1]
                if len(layer_weights.shape) == 2:
                    # If layer_weights is 2-dimensional, need to apply weighted sum for each weight update
                    new_error = np.array([np.dot(row, last_error) for row in layer_weights])
                    new_w = layer_weights + np.matmul(new_error, layer_weights) * self.learning_rate
                    new_weights.append(new_w)
                    errors.append(new_error)
                else:
                    # Intermediary rows that are not the last layer do
                    # not contain any values (I suppose not used by keras?)
                    new_weights.append(layer_weights)
        # layer weights should be reversed so input layer is first, output layer is last
        new_weights = new_weights[::-1]
        self.embedder.set_weights(new_weights)
        return new_weights


class SignMatrix(Matrix):
    """
    Represents a matrix consisting of 1/0's
    A point is equal to ysign(F + G)
    """

    def __init__(self, matrix_f: EmbeddingMatrix, matrix_g: EmbeddingMatrix, gamma=1):
        assert (matrix_f.matrix.shape == matrix_g.matrix.shape)
        super().__init__()
        self.gamma = gamma
        self.matrix_f = matrix_f
        self.matrix_g = matrix_g
        self.matrix = np.empty(shape=matrix_f.matrix.shape)
        self.recalculate()

    def recalculate(self):
        self.matrix = self.gamma * np.sign(self.matrix_f.matrix + self.matrix_g.matrix)
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
        self.matrix = 1 / 2 * np.matmul(self.G.matrix, np.transpose(self.F.matrix))


class SimilarityMatrix(Matrix):
    """
    This matrix represents the similarity of images and captions
    [i,j] == 1 if image at index i and caption at index j should be similar (are a pair)
    """

    def __init__(self, pairs, nr_images, nr_captions):
        super().__init__()
        self.matrix = np.zeros(shape=(nr_images, nr_captions))
        for (index_image, index_caption) in pairs:
            self.matrix[index_image, index_caption] = 1  # value is 1 if pair exists AKA image and caption are similar
