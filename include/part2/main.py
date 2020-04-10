import numpy as np

from include.part2.embedding.embedding import get_image_embedder, get_caption_embedder
from include.part2.embedding.matrix import EmbeddingMatrix, SignMatrix

"""
Parameters:
"""
gamma = 1
eta = 1
c = 32

"""
Training loop:

1. Create neural networks for image and caption embedding
2. initialize B as ysign(F + G)
3. update weights of F based on custom loss
4. update weights of G based on custom loss
5. update B as ysign(F + G)

"""

image_embedder = get_image_embedder(2048)
caption_embedder = get_caption_embedder(4096)

# dummy values to test with
x = np.random.rand(100, 2048) * 2 - 1
y = np.random.rand(100, 4096) * 2 - 1

# Create embedding matrices
F = EmbeddingMatrix(embedder=image_embedder, datapoints=x)
G = EmbeddingMatrix(embedder=caption_embedder, datapoints=y)

# Create sign matrix
B = SignMatrix(matrix_f=F, matrix_g=G)
