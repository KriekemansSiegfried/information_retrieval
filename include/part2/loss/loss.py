from math import exp
import numpy as np


def weight_factor(row, column, theta_matrix):
    """
    return factor scaling for loss function
    :return: 
    """
    theta_val = theta_matrix[row, column]
    return 1 / (1 - exp(-1 * theta_val))


def f_loss(nr_of_samples, nr_of_pairs, theta_matrix, F, G, B, S, gamma=1, eta=1):
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
    for i in range(nr_of_samples):
        # calculate first factor
        factor1 = 0
        for j in range(nr_of_pairs):
            first_val = weight_factor(i, j, theta_matrix) * G[:, j]
            second_val = S[i, j] * G[:, j]
            factor1 += (first_val - second_val)
        factor1 /= 2

        # calculate second factor
        factor2 = 2 * gamma * (F[:, i] - B[:, i])

        # calculate third factor
        factor3 = 2 * eta * np.dot(F, L)

        loss_val += (factor1 + factor2 + factor3)
    return loss_val


def g_loss(nr_of_samples, nr_of_pairs, theta_matrix, F, G, B, S, gamma=1, eta=1):
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
    for j in range(nr_of_samples):
        # calculate first factor
        factor1 = 0
        for i in range(nr_of_pairs):
            first_val = weight_factor(i, j, theta_matrix) * F[:, i]
            second_val = S[i, j] * F[:, i]
            factor1 += (first_val - second_val)
        factor1 /= 2

        # calculate second factor
        factor2 = 2 * gamma * (G[:, j] - B[:, j])

        # calculate third factor
        factor3 = 2 * eta * np.dot(G, L)

        loss_val += (factor1 + factor2 + factor3)
    return loss_val
