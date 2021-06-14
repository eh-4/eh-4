# -*- coding: utf-8 -*-
"""Machine Learning Exercise 06 - SVM Kernel
This exercise covers gridsearch with an SVM, adding kernels to SVMs,
and an example from sklearn covering what different kernels do.
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # This is the SVM classifier

plt.style.use("ggplot")

## Generate and plot synthetic data
X, y = make_blobs(n_samples=1000,random_state=0)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.title("3 classes of data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

print(f"All Data:        {len(X)} points")
print(f"Training data:   {len(X_train)} points")
print(f"Testing data:    {len(X_test)} points")

# Create set of hyperparameters to grid search. We will only vary C for the sake of time
hyperparams = {
    "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4],
    "random_state": [0]
}

# Create a GridSearchCV with SVC, the provided hyperparameters, and use the accuracy scoring method.
svc = SVC()
clf = GridSearchCV(svc, hyperparams, scoring="accuracy", cv=5)

# Fit the clf estimator to the training data
clf.fit(X_train,y_train)

# Extract the best estimator, best score and best parameters
clf.best_estimator_
clf.best_score_
clf.best_params_

assert clf.best_score_ > 0.9

# Use clf.score to get the performance metric on the test data
testScore =  clf.score(X_test, y_test)

assert testScore > 0.9

# Kernel SVM
# Set up data for kernel svm
circlesX, circlesY = make_circles(300, noise=0.1, random_state=0, factor=0.1)

plt.scatter(circlesX[:, 0], circlesX[:, 1], marker='o', c=circlesY, s=25, edgecolor='k')
plt.title("2 Concentric Circles")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Hyperparameters for Kernel SVMs
circleParams = {
    "kernel": ["linear", "poly"],
    "degree": [1, 2, 3],
    "C": [1e-2, 1e-1, 1, 1e2],
    "random_state": [0],
}

# Split data into train, test
circles_X_train, circles_X_test, circles_y_train, circles_y_test = train_test_split(circlesX, circlesY, random_state=0, test_size=0.2)

print(f"All Data:        {len(circlesX)} points")
print(f"Training data:   {len(circles_X_train)} points")
print(f"Testing data:    {len(circles_X_test)} points")

# Create the gridsearchCV kernel SVM classifier
circlesSvc = SVC()
circlesClf = GridSearchCV(circlesSvc, circleParams, scoring="f1_weighted")
circlesClf.fit(circles_X_train,circles_y_train)

# Output
circlesClf.best_estimator_
circlesScore = circlesClf.best_score_
circlesClf.best_score_

assert circlesScore > 0.9

## CODE FROM https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
# Shows how different kernels work
def make_meshgrid(x, y, h=.02):
    # Creates a mesh of points to plot in
    # Inputs:
    #   x: data to base x-axis meshgrid on
    #   y: data to base y-axis meshgrid on
    #   h: stepsize for meshgrid, optional
    # Returns:
    #   xx, yy : ndarray

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    # Plots the decision boundaries for a classifier.
    # Inputs:
    #   ax: matplotlib axes object
    #   clf: a classifier
    #   xx: meshgrid ndarray
    #   yy: meshgrid ndarray
    #   params: dictionary of params to pass to contourf, optional

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# import some data to play with
iris = sk.datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# Create an instance of SVM and fit out data. We do not scale our
# input data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (sk.svm.SVC(kernel='linear', C=C),
          sk.svm.LinearSVC(C=C),
          sk.svm.SVC(kernel='rbf', gamma=0.7, C=C),
          sk.svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()