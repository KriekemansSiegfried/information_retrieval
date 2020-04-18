from math import exp
import numpy as np


def weight_factor(row, column, theta_matrix):
    """
    return factor scaling for loss function
    :return: 
    """
    theta_val = theta_matrix[row, column]
    return 1 / (1 + exp(-1 * theta_val))


def f_loss(samples, all_pairs, theta_matrix, F, G, B, S, gamma=1, eta=1):
    """
    function that calculates the loss of F as written in the assignment
    This function serves as a loss to update the weights of the embedders
    :param nr_of_samples: samples that have been converted to embedding
    :param nr_of_pairs : nr of pairs in the dataset
    :param theta_matrix:  1/2 * F^T * G
    :param F: embedding matrix of images
    :param G: embedding matrix of captions
    :param B:  matrix with binary hashes
    :param S: similarity matrix (images x captions)
    :param gamma: scaling factor -> 1 as in assignment
    :param eta : scaling factor -> 1 as in assignment
    :return: loss as written out in the assignment
    """
    loss_val = 0
    L = np.ones(F.shape[1])
    for (image_index, caption_index) in samples:
        # calculate first factor
        factor1 = 0
        for (pair_image_index, pair_caption_index) in all_pairs:
            first_val = weight_factor(image_index, pair_caption_index, theta_matrix) * G[:, pair_caption_index]
            second_val = S[image_index, pair_caption_index] * G[:, pair_caption_index]
            factor1 += (first_val - second_val)
        factor1 /= 2

        # calculate second factor
        factor2 = 2 * gamma * (F[:, image_index] - B[:, image_index])

        # calculate third factor
        factor3 = 2 * eta * np.matmul(F, L)

        loss_val += (factor1 + factor2 + factor3)
    return loss_val


def g_loss(samples, all_pairs, theta_matrix, F, G, B, S, gamma=1, eta=1):
    """
    function that calculates the loss of G as written in the assignment
    This function serves as a loss to update the weights of the embedders
    :param nr_of_samples: samples that have been converted to embedding
    :param nr_of_pairs : nr of pairs in the dataset
    :param theta_matrix:  1/2 * F^T * G
    :param F: embedding matrix of images
    :param G: embedding matrix of captions
    :param B:  matrix with binary hashes
    :param S: similarity matrix (images x captions)
    :param gamma: scaling factor -> 1 as in assignment
    :param eta : scaling factor -> 1 as in assignment
    :return: loss as written out in the assignment
    """
    L = np.ones(G.shape[1])
    loss_val = 0
    for (image_index, caption_index) in samples:
        # calculate first factor
        factor1 = 0
        for (pair_image_index, pair_caption_index) in all_pairs:
            first_val = weight_factor(pair_image_index, caption_index, theta_matrix) * F[:, pair_image_index]
            second_val = S[pair_image_index, caption_index] * F[:, pair_image_index]
            factor1 += (first_val - second_val)
        factor1 /= 2

        # calculate second factor
        factor2 = 2 * gamma * (G[:, caption_index] - B[:, caption_index])

        # calculate third factor
        factor3 = 2 * eta * np.matmul(G, L)

        loss_val += (factor1 + factor2 + factor3)
    return loss_val
