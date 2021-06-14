# -*- coding: utf-8 -*-
"""Machine Learning Exercise 07 - KMeans & Metrics for Unsupervised Learning - Emerson Ham
This exercise covers cases of unsupervised learning clustering where the actual labels are known, but witheld.
kmeans is used and several metrics are used to test performance
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, silhouette_score
import inspect

# %matplotlib inline
plt.style.use("ggplot")

# Create synthetic data
n = 10000
X, y = make_blobs(n_samples=n, centers=20, n_features=2, random_state=0)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.title("10 classes of data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Create a KMeans model, fit it, and predict clusters
model = KMeans(n_clusters=20, random_state=0).fit(X)
y_pred = model.predict(X)

# Plot the fitted centroids on top of the data
centroids = model.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", c="r")
plt.title("10 classes of data with cluster centroids")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

def contingency_table(y_1, y_2):
    # Determines the contingency matrix based on two sets of element groupings
    # Inputs:
    #   y_1 (iterable): A labeling of elements into clusters
    #   y_2 (iterable): Another labeling of elements into clusters
    # Returns:
    #  np.ndarray: A matrix with shape n_groups_y1, n_groups_y2 listing the number of
    #              elements from one cluster in y1 that are also in one cluster in y2

    conMat = sk.metrics.cluster.contingency_matrix(y_1, y_2, eps=None, sparse=False)
    return conMat

# Use function to create contingency table
test_cont = contingency_table(np.array([0,0,1,2,2]), np.array([1,1,1,0,2]))
assert test_cont.shape == (3,3)
assert np.all(test_cont.T == contingency_table(np.array([1,1,1,0,2]), np.array([0,0,1,2,2])))

# Output
contingency_table(y, y_pred)
contingency_table(y, y)
contingency_table(y_pred, y_pred)

# Other metrics
# The Adjusted Rand Index is a measure of similarity between two sets of labels that normalizes for randomness. Perfect labeling results in a score of 1 and poor labeling results in a negative or 0 score.
print(f"Our model has an adjusted rand index of {adjusted_rand_score(y, y_pred):.4f}")

# Adjusted Mutual Information is another measure of consensus. It measures the agreement between two sets after normalizing for chance. Similar to ARI, perfect labeling gets a score of 1 and poor labeling results in a negative or 0 score."""
print(f"Our model has an adjusted mutual information score of {adjusted_mutual_info_score(y, y_pred, average_method='arithmetic'):.4f}")

# Homogeneity is a measure of clusters containing only points from the same class
# Completeness is a measure of clusters containing all points of a particular class
# V-Measure is the hyperbolic mean of the two
# All 3 are bounded between 0 and 1 where 1 is a perfect score.
h,c,v = homogeneity_completeness_v_measure(y, y_pred)
print(f"Our model has a homogeneity score of {h:.4f}, completeness score of {c:.4f}, and a v-measure of {v:.4f}.")

# Lets see how K affects silhouette score
def get_silhouette_ks(ks, features):
    # Get silhoutte scores of applying KMeans to a set of features with a variety of k values
    # Inputs:
    #   ks (iterable): List of k values to experiment with
    #   features (iterable): The features to train the model on
    # Returns:
    #   iterable: A list of scores of models trained using each value of k

    scores = []
    for k in ks:
      # test
      model = KMeans(n_clusters=k, random_state=0).fit(features)
      y_pred = model.predict(features)
      #calculate silhouette score from data and predicted labels
      scores.append(sk.metrics.silhouette_score(features, y_pred, metric='euclidean', sample_size=None, random_state=None))
    return scores

# Get silhouette scores
ks = [2,3,4,5,6,10,20,50, 100]
scores = get_silhouette_ks(ks, X)

# Output
maxScore = 0
bestK = ks[0]
for k, score in zip(ks, scores):
    if score > maxScore:
        bestK = k
        maxScore = score
    print(f"With k = {k}, the silhoute score is {score:.4f}")
    
print(f"\n\nThe best k value is k={bestK}, scoring {maxScore:.4f}.")

# Plot scores versus k
plt.plot(ks, scores)
plt.xlabel("k")
plt.ylabel("silhoutte score")
plt.title("Silhoutte scores of KMeans clustering")
plt.show()

plt.plot(ks[:6], scores[:6])
plt.xlabel("k")
plt.ylabel("silhoutte score")
plt.title("Subset of Silhoutte scores of KMeans clustering")
plt.show()