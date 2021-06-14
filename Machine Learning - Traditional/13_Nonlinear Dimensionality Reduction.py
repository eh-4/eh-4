# -*- coding: utf-8 -*-
"""Machine Learning Exercise 13 - Dimensionality Reduction - Emerson Ham
T-SNE, LDA, and Spectral embeddings are compared
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import scipy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

np.random.seed(0)
plt.style.use("ggplot")

# Load breast cancer data from sklearn and convert to dataframes
data = load_breast_cancer()

# Plot into on data
features = pd.DataFrame(data["data"], columns=data["feature_names"])
target = pd.Series(data["target"], name="class")
print(data["DESCR"])

features.head()
features.describe()
target.head()

# Split data into train, test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# T-SNE embedding visualization (often used to visualize high dof data)
perplexities = [5, 20, 30, 50, 100]
iters = [250, 1000, 3000]

tsne_embeddings = []
for perplexity in perplexities:
    fig, axes = plt.subplots(nrows=1, ncols=len(iters), figsize=(16, 8), sharex=True, sharey=True)
    for i,n in enumerate(iters):
        # Calculate the tsne_embedding using the perplexity and n values and a random_state of 0
        # YOUR CODE HERE
        tsne_embedding = TSNE(n_components=2, perplexity=perplexity, n_iter=n).fit_transform(X_train)
        
        axes[i].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=y_train)
        axes[i].set_title(f"t-SNE\nPerplexity={perplexity}, {n} steps")
    tsne_embeddings.append(tsne_embedding)
    plt.show()

# Spectral Embedding visualization
spectral = SpectralEmbedding(n_components=2)
spectral_embedding = spectral.fit_transform(X_train)

plt.scatter(spectral_embedding[:, 0], spectral_embedding[:, 1], c=y_train)
plt.title("Spectral Embedding")
plt.show()

# Linear Discriminant Analysis(LDA) embedding visualization
lda = LinearDiscriminantAnalysis(n_components=2)
lda_embedding = lda.fit_transform(X_train, y_train)

plt.scatter(lda_embedding, [0]*len(lda_embedding), c=y_train)
plt.yticks([])
plt.title("LDA Embedding")
plt.show()

# Test embeddings with classifier
#Merge Dataframes for t-SNE and Spectral Embedding
X_frames = [X_train, X_test]
X_cct = pd.concat(X_frames)
y_frames = [y_train, y_test]
y_cct = pd.concat(y_frames)

#t-SNE embedding
selectedPerplexity = 30;
X_train_length = X_train.shape[0]
tsne_model = TSNE(n_components=2, perplexity=selectedPerplexity, n_iter=n)
tsne_result = tsne_model.fit_transform(X_cct)
tsne_train = tsne_result[:X_train_length]
tsne_test = tsne_result[-(tsne_result.shape[0]-X_train_length):]

#Spectral Embedding
spectral_model = SpectralEmbedding(n_components=2)
spectral_result = spectral_model.fit_transform(X_cct)
spectral_train = spectral_result[:X_train_length]
spectral_test = spectral_result[-(spectral_result.shape[0]-X_train_length):]

#LDA Embedding
lda_result = lda.fit_transform(X_cct, y_cct)
lda_train = spectral_result[:X_train_length]
lda_test = spectral_result[-(spectral_result.shape[0]-X_train_length):]
print(lda_test.shape)

# Fit SVMs to embedded data
clf = LinearSVC(random_state=0, tol=1e-5)
tsne_svc = clf.fit(tsne_train,y_train)  
lda_svc = clf.fit(lda_train,y_train)  
spectral_svc = clf.fit(spectral_train,y_train)

# Output
print(f"The t-SNE embedding + Linear SVM scores an F-1 = {f1_score(y_test, tsne_svc.predict(tsne_test)):.3f}.")
print(f"The Spectral Embedding + Linear SVM scores an F-1 = {f1_score(y_test, spectral_svc.predict(spectral_test)):.3f}.")
print(f"The LDA + Linear SVM scores an F-1 = {f1_score(y_test, lda_svc.predict(lda_test)):.3f}.")

all_feat_score = f1_score(y_test, LinearSVC().fit(X_train, y_train).predict(X_test))
print(f"The Linear SVM with all features scores an F-1 = {all_feat_score:.3f}.")