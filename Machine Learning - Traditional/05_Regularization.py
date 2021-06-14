# -*- coding: utf-8 -*-
"""Machine Learning Exercise 05 - Regularization - Emerson Ham
In this exercise we're going to look at lowering the chance of overfitting by using regularization parameters.
This will include looking at different norms
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

plt.style.use("ggplot")

# Generate a Hilbert matrix dataset to fit
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# Create some alpha values that we'll use to train several models
n_alphas = 200
alphas = np.logspace(-10, -1, n_alphas)

def determine_coefficients(alphas, model, X, y):
    # Determine the coefficients of a linear model (Lasso or Ridge) given the various alphas.
    # You should train a model for each value of alpha and store its coefficients to be returned.
    # Args:
    #   alphas (iterable): The alphas to test out with the model
    #   model (sklearn.estimator Class): A type of linear model not instantiated
    #   X (iterable): The data to train on
    #   y (iterable): The labels to train on
    # Returns:
    #   coefs (iterable): the coefficients extracted from the trained model. See model.coef_

    coefs = []
    model.fit_intercept = False
    for alpha in alphas:
        model.alpha = alpha
        model.fit(X,y)
        coefs.append(model.coef_)
        
    return coefs

# For a Ridge Regression model, use function to determine the coefficients for each alpha
coefs = determine_coefficients(alphas, Ridge(), X, y)

# Plot the coefficients versus alpha
# This shows how Ridge Regression makes the coefficients smaller
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficient Value')
plt.title('Ridge coefficients as a \nfunction of the regularization weight')
plt.axis('tight')
plt.show()


# For a Lasso Regression model, use function to determine the coefficients for each alpha
lassoCoefs = determine_coefficients(alphas, Lasso(), X, y)

# Plots the coefficients versus alpha (note the coefficinets are significantly smaller)
# This shows how Lasso regression forces coefficients to zero
ax = plt.gca()
ax.plot(alphas, lassoCoefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficient Value')
plt.title('Lasso coefficients as a \nfunction of the regularization weight')
plt.axis('tight')
plt.show()

## For an Elastic Net Model(combination of lasso and ridge), get the coefficients for varied alpha and L1 values
l1_ratios = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

model = ElasticNet(random_state=0)
elastic_coefs = {}
model.fit_intercept = False
for l1_ratio in l1_ratios:
    elastic_coefs[l1_ratio] = []
    model.l1_ratio = l1_ratio
    for alpha in alphas:
        # Create an Elastic net model with the alpha and l1_ratio provided
        # Save that model to the variable model
        # Add the model's coefficients to elastic_coefs at the l1_ratio key's list
        # YOUR CODE HERE
        model.alpha = alpha
        model.fit(X,y)
        elastic_coefs[l1_ratio].append(model.coef_)

# Plot the coefficinets versus alpha for each l1 value
# This shows how l1 is the ratio between shrinking coefficients and driving them to zero
for l1_ratio in l1_ratios:
    ax = plt.gca()
    ax.plot(alphas, elastic_coefs[l1_ratio])
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('Coefficient Value')
    plt.title(f'Elastic Net Coefficients\nl1_ratio={l1_ratio}')
    plt.axis('tight')
    plt.show()


