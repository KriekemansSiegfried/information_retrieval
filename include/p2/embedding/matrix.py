import numpy as np


class EmbeddingMatrix:
    """
    Represents a matrix that contains the embeddings of a series of datapoints
    """

    def __init__(self, embedder, datapoints):
        self.embedder = embedder
        self.embedding_size = embedder.layers[-1].output.shape[1]
        self.datapoints = datapoints
        self.matrix = np.empty(shape=(datapoints.shape[0], self.embedding_size))
        self.recalculate()

    def update(self, samples):
        """ update weights of embedder """
        pass

    def recalculate(self):
        """ recalculate the internal matrix based on the embeddings of datapoints"""
        self.matrix = self.embedder.predict(self.datapoints)


class SignMatrix:
    """
    Represents a matrix consisting of 1/0's
    A point is equal to ysign(F + G)
    """

    def __init__(self, matrix_f: EmbeddingMatrix, matrix_g: EmbeddingMatrix, gamma=1):
        assert (matrix_f.matrix.shape == matrix_g.matrix.shape)
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


class ThetaMatrix:
    """
    This represents 1/2 * F^T * G
    """

    def __init__(self, F, G):
        self.F = F
        self.G = G
        self.matrix = None
        self.recalculate()

    def recalculate(self):
        self.matrix = 1 / 2 * np.matmul(self.G.matrix, np.transpose(self.F.matrix))


class SimilarityMatrix:

    def __init__(self, pairs, nr_images, nr_captions):
        self.matrix = np.zeros(shape=(nr_images, nr_captions))
        for (index_image, index_caption) in pairs:
            self.matrix[index_image, index_caption] = 1  # value is 1 if pair exists AKA image and caption are similar
