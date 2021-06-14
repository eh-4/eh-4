# -*- coding: utf-8 -*-
"""Machine Learning Exercise 8 - Hierarchical & Spectral(Manifold) Clustering - Emerson Ham
In this exercise we will look at a variety of clustering methods where the labels are not known
This includes hierarchical Spectral(manifold) methods.
"""

import numpy as np
import scipy as sc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, silhouette_score
from sklearn.cluster import SpectralClustering

plt.style.use("ggplot")

# We used the [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) for this exercise.
iris = load_iris()
iris_features = iris["data"]
iris_targets = iris["target"]
print(iris["DESCR"])

# Agglomerative Clustering
def dendrogram_plotter(features, methods, metric):
    # Plots a dendrogram for the provided features for every method in methods using the provided metric
    # Inputs:
    #   features (iterable): The features to use in creating the dendrogram
    #   methods (iterable): A list of strings where each one is a valid method to the linkage function
    #   metric (str): A metric for calculating the linkage

    for method in methods:
        plt.figure(figsize = (10,6)) # Change the figure size to your liking
        # YOUR CODE HERE
        linkageMatrix = sc.cluster.hierarchy.linkage(features, method=method, metric=metric, optimal_ordering=False)
        sc.cluster.hierarchy.dendrogram(linkageMatrix, p=30, truncate_mode=None, color_threshold=None, get_leaves=True, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True, no_plot=False, no_labels=False, leaf_font_size=None, leaf_rotation=None, leaf_label_func=None, show_contracted=False, link_color_func=None, ax=None)
        plt.title(f"{method.title()} Linkage Iris Dataset Dendrogram")
        plt.show()

# Plot dendogram
dendrogram_plotter(iris_features, ["average", "complete", "ward"], "euclidean")

def clustering_scorer(features, targets, pred):
    # Calculates all the important clustering scores given a set of features, targets and predictions
    # Inputs
    #   features (iterable): The input features to the clustering problem
    #   targets (iterable): The targets if this was a classification problem
    #   pred (iterable): The cluster predictions for the data samples
    # Returns:
    #   dict: A dictionary with the keys ['Adjusted Rand', 'Adjusted Mutual Info', 'Homogeneity', 'Completeness',
    #       'V Score', 'Silhouette Score'] and values as the respective scores for the inputs.

    scores = {}
    scores['Adjusted Rand'] = sk.metrics.adjusted_rand_score(targets, pred)
    scores['Adjusted Mutual Info'] = sk.metrics.adjusted_mutual_info_score(targets, pred, average_method='warn')
    scores['Homogeneity'] = sk.metrics.homogeneity_score(targets, pred)
    scores['Completeness'] = sk.metrics.completeness_score(targets, pred)
    scores['V Score'] = sk.metrics.v_measure_score(targets, pred, beta=1.0)
    scores['Silhouette Score'] = sk.metrics.silhouette_score(targets.reshape(-1, 1), pred, metric='euclidean')
      #^This line produces the error, the previous ones do not, suggesting targets and pred are correct
    return scores

# Agglomerative clustering
def agg_clustering_scorer(features, targets, linkages, n_clusters=8):
    # Calculate the agglomerative clustering scores for a variety of linkage types
    # Inputs
    #   features (iterable): The input features of the data
    #   targets (iterable): The target classes if this was treated as a classification problem
    #   linkages (iterable): A list of linkage methods to calculate scores for
    #   n_clusters (int, optional): Defaults to 8. The number of clusters to use in the clustering algorithm
    # Returns:
    #   iterable: Scores for each linkage method similar to the clustering_scorer method's output

    scores = [dict() for x in range(sum(1 for e in linkages))]
    idx = 0
    for linkage in linkages:
      # YOUR CODE HERE
      pred = sk.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(features)
      thisScore = clustering_scorer(features,targets,pred.labels_)
      scores[idx] = thisScore
      idx+=1
    return scores

aggScores = agg_clustering_scorer(iris_features, iris_targets, ["average", "complete", "ward"], n_clusters=3)

# Compare linkage metrics
for linkage, score in zip(["average", "complete", "ward"],aggScores):
    print(f"With the {linkage} linkage,")
    print(f"Adjusted rand score is {score['Adjusted Rand']}")
    print(f"Adjusted mutual info score is {score['Adjusted Mutual Info']}")
    print(f"Homogeneity is {score['Homogeneity']}, Completeness is {score['Completeness']}, V score is {score['V Score']}")
    print(f"Silhouette score is {score['Silhouette Score']}\n")

scoresdf = pd.DataFrame(aggScores, index=["average", "complete", "ward"]).T
scoresdf.plot(kind="barh")
plt.title("Clustering scores with various linkage methods")
plt.show()

# Spectral Clustering
# Create, fit and predict a spectral clustering classifier
clf = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0)
spectral_pred = clf.fit(iris_features).labels_

# Get and plot scores
spectral_scores = clustering_scorer(iris_features, iris_targets, spectral_pred)

if len(aggScores) == 3:
    aggScores.append(spectral_scores)
scoresdf = pd.DataFrame(aggScores, index=["average", "complete", "ward", "spectral"]) # Try removing the .T and see what happens
scoresdf.plot(kind="barh")
plt.title("Clustering scores with various linkage methods and Spectral Clustering")
plt.xlim(0,1.5)
plt.show()

# Same code copied from http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
# in exercise 07 since we've now covered some more of these methods

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()