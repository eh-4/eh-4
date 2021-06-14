# -*- coding: utf-8 -*-
"""14 - Deep Learning.ipynb
This exercise uses a basic Keras FFNN to predict house prices
Adam optimizer and dropout are added
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from keras import metrics
from tensorflow.keras.layers import Dense, Dropout, Activation

np.random.seed(0)

# Load keras housing data
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

x_train.shape
y_train.shape
y_train.mean(), y_train.std()

# Create layers and convert them into a model using sequential()
layers = [ Dense(10, activation='relu', input_shape=(13,)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='relu') ]
model = keras.Sequential(layers)

# Set up the model to do the following:
# - use stochastic gradient descent to fit the model
# - use mean absolute error as its loss function
# - use mean absolute error as one of its metrics
# - use mean squared error as one of its metrics
model.compile(optimizer='sgd', loss='mae', metrics=["mae", "mse"])
model.summary()

# Now fit the model
model.fit(x_train, y_train, epochs=50)

# Evaluate how the model does based on the test data
model.evaluate(x_test, y_test)

# Switch the optimizer to ADAM
model.compile(optimizer='adam', loss='mae', metrics=['mae', "mse"])
model.summary()
model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)

# Add dropout
layers = [ Dense(10, activation='linear', input_shape=(13,)),
    Dropout(0.5),
    Dense(10, activation='linear'),
    Dropout(0.5),
    Dense(10, activation='linear'),
    Dense(1, activation='linear') ]
model = keras.Sequential(layers)

# Train the updated layers
model.compile(optimizer='adam', loss='mae', metrics=['mae', "mse"])
model.fit(x_train, y_train, epochs=500, verbose=0)
model.summary()

model.evaluate(x_test, y_test)