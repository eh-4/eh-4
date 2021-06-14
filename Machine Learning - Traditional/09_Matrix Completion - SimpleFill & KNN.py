# -*- coding: utf-8 -*-
"""Machine Learning Exercise 09 - Matrix Completion - Emerson Ham
In this exercise we use various matrix completion algorithms to fill the incomplete matrix
"""

#!pip install fancyimpute
#!pipenv install fancyimpute

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from fancyimpute import SimpleFill, KNN, MatrixFactorization
from sklearn.neighbors import KNeighborsClassifier

plt.style.use("ggplot")
np.random.seed(0)

# Create synthetic data matrix
n = 30
m = 30
inner_rank = 5
X = np.dot(np.random.randn(n, inner_rank), np.random.randn(inner_rank, m))

# Plot sythetic matrix
plt.figure()
plt.imshow(X)
plt.grid(False)
plt.show()

# Delete a portion of the data
# Since our values are random from 0 to 1 we can select a cutoff to remove all points less than a value
# Doing so is a good enough approximation of removing that percentage of the matrix
cutoff = 0.4
missing_mask = np.random.rand(*X.shape) < cutoff
X_incomplete = X.copy()
X_incomplete[missing_mask] = np.nan

# Plot Incomplete Matrix
plt.figure()
plt.imshow(X_incomplete)
plt.grid(False)
plt.show()

# Use fancyimpute's simplefill algorithm to replace the missing values
# Simplefill fills the missing points with average values
meanFill = SimpleFill("mean")
X_filled_mean = meanFill.fit_transform(X_incomplete)

# Plot results
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
ax1.imshow(X)
ax1.set_title("Original Matrix")
ax1.grid(False)

ax2.imshow(X_filled_mean)
ax2.set_title("Mean Fill Completed Matrix")
ax2.grid(False)

ax3.imshow(X_incomplete)
ax3.set_title("Incomplete Matrix")
ax3.grid(False)
plt.show()

# MSE Metric
def mat_completion_mse(X_filled, X_truth, missing_mask):
    # Calculates the mean squared error of the filled in values vs. the truth
    # Inputs:
    #   X_filled (np.ndarray): The "filled-in" matrix from a matrix completion algorithm
    #   X_truth (np.ndarray): The true filled in matrix
    #   missing_mask (np.ndarray): Boolean array of missing values
    # Returns:
    #   float: Mean squares error of the filled values

    mse = ((X_filled[missing_mask] - X[missing_mask]) ** 2).mean()
    return mse

meanFill_mse = mat_completion_mse(X_filled_mean, X, missing_mask)
print("meanFill MSE: %f" % meanFill_mse)

# KNN Matric Completion
def find_best_k(k_neighbors, complete_mat, incomplete_mat, missing_mask):
    # Determines the best k to use for matrix completion with KNN
    # Args:
    #   k_neighbors (iterable): The list of k's to try
    #   complete_mat (np.ndarray): The original matrix with complete values
    #   incomplete_mat (np.ndarray): The matrix with missing values
    #   missing_mask (np.ndarray): Boolean array of missing values
    # Returns:
    #   integer: the best value of k to use for that particular matrix

    best_k = -1
    best_k_mse = np.infty
    
    for neighbors in k_neighbors:
        # YOUR CODE HERE
        X_filled = KNN(k=neighbors).fit_transform(incomplete_mat)
        this_k_mse = mat_completion_mse(X_filled, complete_mat, missing_mask)
        if this_k_mse < best_k_mse:
            best_k_mse = this_k_mse
            best_k = neighbors
    return best_k

# Find best number k of neighbors for the KNN algorithm to use
k_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
best_k = find_best_k(k_neighbors, X, X_incomplete, missing_mask)

# Run KNN with the best_k and calculate the mean squared error
X_filled_knn = KNN(k=best_k).fit_transform(X_incomplete)
knnFill_mse = mat_completion_mse(X_filled_knn, X, missing_mask)
print("knnFill MSE: %f" % knnFill_mse)

# Alternating Minimizaiton based methods from the handout in class
class AlternatingMinimization:
    def __init__(self,rank):
        self.rank = rank
        
    def fit_transform(self, X_incomplete):
        # Fits and transforms an incomplete matrix, returning the completed matrix.

        P = np.random.random_sample((n, self.rank))
        Q = np.random.random_sample((self.rank, m))
        # Fill in all missing values with zeros
        X_incomplete_to_zero = np.nan_to_num(X_incomplete)

        for i in range(0, 100):
            P = X_incomplete_to_zero @ Q.T @ np.linalg.pinv(Q @ Q.T)
            Q = np.linalg.pinv(P.T @ P) @ P.T @ X_incomplete_to_zero

        X_filled = P @ Q
        
        return X_filled

# Visual Comparison of the 4 methods:
#   Mean Fill
#   K-Nearest Neighbors
#   Alternating Minimization
# MatrixFactorization (an implementaiton using gradient descent)
simpleFill = SimpleFill("mean")
knnFill = KNN(k=best_k)
amFill = AlternatingMinimization(rank=5)
mfFill = MatrixFactorization(learning_rate=0.01, rank=5, l2_penalty=0, min_improvement=1e-6)
methods = [simpleFill, knnFill, amFill, mfFill]
names = ["SimpleFill", "KNN", "AltMin", "MatFactor"]

def mat_completion_comparison(methods, incomplete_mat, complete_mat, missing_mask):
    # Using a list of provided matrix completion methods calculate the completed matrix and the determine the associated
    # mean-squared-error results.
    # Inputs
    #   methods (iterable): A list of matrix completion algorithms
    #   incomplete_mat (np.ndarray): The incomplete matrix
    #   complete_mat (np.ndarray): The full matrix
    #   missing_mask (np.ndarray): Boolean array of missing values
    # Returns:
    #   filled_mats (iterable): the "filled-in" matrices
    #   mses (iterable): the mean square error results

    X_filled_mats = []
    mses = []
    for method in methods:
        # YOUR CODE HERE
        pred = method.fit_transform(incomplete_mat)
        X_filled_mats.append(pred)
        mses.append(mat_completion_mse(pred, complete_mat, missing_mask))
        
    return X_filled_mats, mses

# Fill matrices and plot metrics for each method
X_filled_mats, mses = mat_completion_comparison(methods, X_incomplete, X, missing_mask)
plt.figure(figsize = (19,6)) # Change the figure size to your liking

for i in range(0, len(methods)):
    X_filled = X_filled_mats[i]
    mse = mses[i]
    ax = plt.subplot(161 + i)
    ax.imshow(X_filled)
    ax.title.set_text("MSE: " + str(round(mse, 2)))
    ax.set_xlabel(names[i])
    ax.grid(False)
plt.show()

ax = plt.subplot(121)
ax.imshow(X)
ax.set_xlabel("Complete")
ax.grid(False)

ax = plt.subplot(122)
ax.imshow(X_incomplete)
ax.set_xlabel("Incomplete")
ax.grid(False)
    
plt.show()