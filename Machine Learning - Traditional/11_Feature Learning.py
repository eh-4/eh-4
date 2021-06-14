# -*- coding: utf-8 -*-
"""Machine Exercise Exercise 11 - Feature Learning - Emerson Ham
In this exercise we used various feature learning methods to simplify data before performing a regression with it
Feature learning methods used are Filter, Wrapper, Lasso, and feature extraction by PCA
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA

plt.style.use("ggplot")

# The data we'll be working with is the [California housing dataset](http://scikit-learn.org/stable/datasets/index.html#california-housing-dataset)."""
house_data = fetch_california_housing()
print(house_data["DESCR"])

# Put the data into a dataframe and display the info
house_features = pd.DataFrame(house_data["data"], columns=house_data["feature_names"])
house_prices = pd.Series(house_data["target"])

print(house_data)
house_features.describe()

house_prices.head()
house_prices.describe()

# Split data into train, test data
X_train, X_test, y_train, y_test = train_test_split(house_features, house_prices, test_size=0.2, random_state=0)

# Filter Selection Method (based on mutual informaiton score)
# Prescribe desired number of features
k = 4

# Create filter selection transformer
mi_transformer = sk.feature_selection.SelectKBest(score_func=sk.feature_selection.mutual_info_regression, k=k).fit(X_train, y_train) #Select the k best features using the mutual info regression as the dependancy score
mi_X_train = mi_transformer.transform(X_train)
mi_X_test = mi_transformer.transform(X_test)

for feature, importance in zip(house_features.columns, mi_transformer.scores_):
    print(f"The MI score for {feature} is {importance}")

# Create basic regression algorithm
miEst = LinearRegression()
miEst.fit(mi_X_train, y_train)

print(f"The mean squared error when training on the MI selected features is {miEst.score(mi_X_train, y_train)}.")
print(f"When testing on the test data, the mean squared error is {miEst.score(mi_X_test, y_test)}")

# Wrapper Feature Selection
# Use an RFE object
rfeEst = LinearRegression()
rfe_transformer = RFE(rfeEst, k, step=2).fit(X_train,y_train)
rfe_X_train = rfe_transformer.transform(X_train)
rfe_X_test = rfe_transformer.transform(X_test)

rfeEst = LinearRegression()
rfeEst.fit(rfe_X_train, y_train)

print(f"The mean squared error when training on the RFE selected features is {rfeEst.score(rfe_X_train, y_train)}.")
print(f"When testing on the test data, the mean squared error is {rfeEst.score(rfe_X_test, y_test)}")

# Embedded Feature Selection Methods
# Create a LassoCV model trained with 10 alphas and save it to lassoClf
lassoClf = LassoCV(random_state=0).fit(X_train, y_train)

lassoClf.coef_
lassoClf.alpha_
print(f"The mean squared error when training using lasso is {lassoClf.score(X_train, y_train)}.")
print(f"When testing on the test data, the mean squared error is {lassoClf.score(X_test, y_test)}")

# Feature Extraction Methods
# Principal Component Analysis(PCA)
pca_transformer = PCA(n_components=k).fit(X_train)
pca_X_train = pca_transformer.transform(X_train)
pca_X_test = pca_transformer.transform(X_test)

pcaEst = LinearRegression()
pcaEst.fit(pca_X_train, y_train)

print(f"The mean squared error when training using PCA is {pcaEst.score(pca_X_train, y_train)}.")
print(f"When testing on the test data, the mean squared error is {pcaEst.score(pca_X_test, y_test)}")

pca_transformer.get_covariance()
pca_transformer.explained_variance_ratio_