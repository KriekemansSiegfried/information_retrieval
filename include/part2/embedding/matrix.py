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

    # def update(self, samples):
    #    for sample in samples:
