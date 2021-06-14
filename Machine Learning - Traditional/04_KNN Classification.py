"""Machine Learning Exercise 04 - K Nearest Neighbors - Emerson Ham
This exercise covers initializing sklearn's KNN model, training it, and using it to make predictions
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

plt.style.use("ggplot")

# Create synthetic data we will use for training
X, y = make_blobs(n_samples=1000,random_state=0)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.title("3 classes of data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Split data into train, validation, and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
X_model, X_valid, y_model, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.2)

print(f"All Data:        {len(X)} points")
print(f"Training data:   {len(X_train)} points")
print(f"Testing data:    {len(X_test)} points")
print(f"Modeling data:   {len(X_model)} points")
print(f"Validation data: {len(X_valid)} points")

# Initialize the KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Fit model
knn.fit(X_model, y_model)

# Predict classes for new data
validationPredictions = knn.predict(X_valid)

# Evaluate the classification performance of the model using several metrics
print(confusion_matrix(y_valid, validationPredictions))
print(classification_report(y_valid,validationPredictions))
f1_score(y_valid, validationPredictions, average="weighted")

#Define some funcitons
def get_knn_training_scores(ks, model_features, model_labels):
    # Measures the f1-score of KNN models that use different Ks
    # Inputs: 
    #   ks (int iterable): iterable of all the k values to apply
    #   model_features (iterable): the features from the model set to train on
    #   model_labels (iterable): the labels from the model set to train on
    # Returns:
    #   key (dictionary): is the k value and value is the weighted f1_score on the validation set

    key = {}
    for i in ks:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(model_features, model_labels)
        pred_labels = knn.predict(model_features)
        key[i] = f1_score(model_labels,pred_labels, average="weighted")
    
    return key

def get_knn_validation_scores(ks, model_features, model_labels, validation_features, validation_labels):
    # Same as above function, but uses a seperate dataset for validation
    # Measures the f1-score of KNN models that use different Ks
    # Inputs: 
    #   ks (int iterable): iterable of all the k values to apply
    #   model_features (iterable): the features from the model set to train on
    #   model_labels (iterable): the labels from the model set to train on
    #   validation_features (iterable): the features from the model set to validate on
    #   validation_labels (iterable): the labels from the model set to validate on
    # Returns:
    #   key (dictionary): is the k value and value is the weighted f1_score on the validation set
    
    key = {}
    for i in ks:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(model_features, model_labels)
        pred_labels = knn.predict(validation_features)
        key[i] = f1_score(validation_labels,pred_labels, average="weighted")
    
    return key

# Use functions to determine best value for K
ksToTest = [1,3,5,7,10,20,50,100]
training_scores = get_knn_training_scores(ksToTest, X_model, y_model)
validation_scores = get_knn_validation_scores(ksToTest, X_model, y_model, X_valid, y_valid)

#  Plot resulting scores
pd.Series(training_scores, name="Training").plot(kind="line")
pd.Series(validation_scores, name="Validation").plot(kind="line", label="Validation")
plt.legend()
plt.xlabel("k")
plt.ylabel("F1-score")
plt.show()

# Select the best value for k and train model using it
bestK = 5
clf = KNeighborsClassifier(bestK)
clf.fit(X_train, y_train)
testPredictions = clf.predict(X_test)

# Assess performance of optimal model
print("Confusion Matrix: \n")
print(confusion_matrix(y_test, testPredictions))
print("\n\nClassification Report:\n")
print(classification_report(y_test, testPredictions))

assert f1_score(y_test, testPredictions, average="weighted") > 0.9