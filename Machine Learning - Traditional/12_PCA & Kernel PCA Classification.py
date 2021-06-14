# -*- coding: utf-8 -*-
"""12 - FL - PCA.ipynb
This exercise uses PCA to extract features about wines and classify them using knn
Kernel PCA is used as well
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles

np.random.seed(0)
plt.style.use("ggplot")

# Download wine data from sklearn and convert to dataframes
wine_data = load_wine()
wine_features = pd.DataFrame(wine_data["data"], columns=wine_data["feature_names"])
wine_targets = pd.DataFrame(wine_data["target"], columns=["class"])

# Display data info
print(wine_data["DESCR"])
wine_features.head()
wine_features.describe()
wine_targets.head()
print(wine_targets["class"].value_counts())
wine_targets["class"].value_counts().plot(kind="barh")

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(wine_features, wine_targets, test_size=0.2, random_state=0)

# Set create PCA transformer and extract the features from the train and test data
pca_tf = PCA(n_components=2).fit(X_train)
X_train_pca = pca_tf.transform(X_train)
X_test_pca = pca_tf.transform(X_test)

# Print how much of the variance is explained by the extracted features
pca_tf.explained_variance_ratio_

plt.scatter(X_train_pca[:,0].reshape(-1,1)[y_train == 0], X_train_pca[:,1].reshape(-1,1)[y_train == 0], color="r")
plt.scatter(X_train_pca[:,0].reshape(-1,1)[y_train == 1], X_train_pca[:,1].reshape(-1,1)[y_train == 1], color="g")
plt.scatter(X_train_pca[:,0].reshape(-1,1)[y_train == 2], X_train_pca[:,1].reshape(-1,1)[y_train == 2], color="b")
plt.show()

# Fit normal KNN classifier and a seperate KNN Classifier with the PCA data
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_pca = KNeighborsClassifier(n_neighbors=10)
knn_pca.fit(X_train_pca, y_train)

# Output the accuracy
print(f"The accuracy using kNN is {accuracy_score(y_test, knn.predict(X_test))}.")
print(f"The accuracy using kNN with PCA is {accuracy_score(y_test, knn_pca.predict(X_test_pca))}.")

# Kernel PCA
# Prep concentric circle data
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
reds = y == 0
blues = y == 1

# Plot data
plt.scatter(X[reds, 0], X[reds, 1], c="r", s=20)
plt.scatter(X[blues, 0], X[blues, 1], c="b", s=20)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# Apply normal PCA
pca_tf = PCA(n_components=2).fit(X)
X_pca = pca_tf.transform(X)

# Plot normal PCA classes to show that it sucks
plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="r", s=20)
plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="b", s=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Apply kernel PCA
kpca_tf = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True).fit(X)
X_kpca = kpca_tf.transform(X)

# Plot the data with kernel PCA applied and classes labeled to show how well it works
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="r", s=20)
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="b", s=20)
plt.xlabel("Kernel PC1")
plt.ylabel("Kernel PC2")
plt.show()

# untransform data and plot classes to show that the original data has been properly classified
X_inverse = kpca_tf.inverse_transform(X_kpca)
plt.scatter(X_inverse[reds, 0], X_inverse[reds, 1], c="r", s=20)
plt.scatter(X_inverse[blues, 0], X_inverse[blues, 1], c="b", s=20)
plt.xlabel("Returned $x_1$ ")
plt.ylabel("Returned $x_2$")
plt.title("The inverse transformed original dimensions")
plt.show()