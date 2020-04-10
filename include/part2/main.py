import numpy as np
from numpy.random import rand, randint

from include.p2.embedding.embedding import get_caption_embedder, get_image_embedder
from include.p2.embedding.matrix import EmbeddingMatrix, SignMatrix, SimilarityMatrix, ThetaMatrix
from include.p2.loss.loss import f_loss, g_loss

# disable_eager_execution()

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
images = rand(1000, 2048)  # TODO: fill in real values
captions = rand(1000, 4096)  # TODO: fill in real values
nr_images = len(images)
nr_captions = len(captions)
image_caption_pairs = [(randint(0, nr_images), randint(0, nr_captions))
                       for i in range(256)]  # TODO: fill in real values

# Create embedding matrices
F = EmbeddingMatrix(embedder=image_embedder, datapoints=images)
G = EmbeddingMatrix(embedder=caption_embedder, datapoints=captions)

# Create theta matrix
theta = ThetaMatrix(F, G)

# get data

# Create similarity matrix
S = SimilarityMatrix(image_caption_pairs, nr_images, nr_captions)

# Create sign matrix
B = SignMatrix(matrix_f=F, matrix_g=G)

# take samples
batch_size = 32

for i in range(0, len(image_caption_pairs), batch_size):
    batch_interval = (i, min(len(image_caption_pairs), i + batch_size))
    print('batch: {}'.format(batch_interval))
    indices = np.arange(batch_interval[0], batch_interval[1])

    image_batch = images[indices, :]
    caption_batch = captions[indices, :]

    image_embeddings = image_embedder.predict(image_batch)
    caption_embeddings = caption_embedder.predict(caption_batch)

    np.put(a=F.matrix, ind=indices, v=image_embeddings)
    np.put(a=G.matrix, ind=indices, v=caption_embeddings)

    loss_f = f_loss(nr_of_samples=len(indices), nr_of_pairs=len(image_caption_pairs),
                    theta_matrix=theta.matrix, F=F.matrix,
                    G=G.matrix, B=B.matrix, S=S.matrix)

    loss_g = g_loss(nr_of_samples=len(indices), nr_of_pairs=len(image_caption_pairs),
                    theta_matrix=theta.matrix, F=F.matrix,
                    G=G.matrix, B=B.matrix, S=S.matrix)

    # image_gradients = gradients(loss_f, image_embedder.trainable_weights)
    # caption_gradients = gradients(loss_g, caption_embedder.trainable_weights)

    B.recalculate()

    print('f loss ({})-> [0] = {}'.format(loss_f.shape, loss_f[0]))
    print('g loss ({}) -> [0] = {}'.format(loss_g.shape, loss_g[0]))

print('samples generated')
