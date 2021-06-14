# -*- coding: utf-8 -*-
"""Machine Learning Exercise 15 - Autoencoders - Emerson Ham
This exercise builds an autoencoder and compare it with PCA in trying to classify data using SVMs.
We use keras's functional model construction method instead of sequential
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Model, Input
import sklearn as sk
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import numpy as np

np.random.seed(0)

# Find GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Load breast cancer data from sklearn
data = load_breast_cancer()
features = data["data"]
targets = data["target"]
X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=0)

# Output info on data
print(data["DESCR"])
X_train.shape
X_train
y_train.shape
y_train

# Determine the input dimensions of each datapoint and use that to create an input object
from tensorflow.keras import layers
embedding_dim = 5
input = keras.Input(shape=(X_train.shape[1],), name='img')
print(X_train.shape[1])

# Create the encoder with a dense hidden layer with a leakyRelu activation function, and another dense layer
x = Dense(10)(input)
x = LeakyReLU(alpha=0.05)(x)
encoder_layers = Dense(embedding_dim)(x)

# Add the decoder with a dense leakyrelu layer and another dense layer
x = Dense(10)(encoder_layers)
x = LeakyReLU(alpha=0.05)(x)
x = Dense(X_train.shape[1])(x)
autoencoder_layers = LeakyReLU(alpha=0.05)(x)

# Create the autoencoder model from the input and layers
autoencoder = Model(input,autoencoder_layers)

# Create the encoder which takes the same inputs but stops at the encoder layers
encoder = Model(input, encoder_layers)

# Create the decoder which starts at the encoder output and uses the remaining layers
embedding = Input(shape=(embedding_dim,))

decoder_layer1 = autoencoder.layers[-4]
decoder_layer2 = autoencoder.layers[-3]
decoder_layer3 = autoencoder.layers[-2]
decoder_layer4 = autoencoder.layers[-1]

decoder_out = decoder_layer1(embedding)
decoder_out = decoder_layer2(decoder_out)
decoder_out = decoder_layer3(decoder_out)
decoder_out = decoder_layer4(decoder_out)

decoder = Model(embedding, decoder_out)

# Compile and train the autoencoder model
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train, X_train, epochs=1000)
autoencoder.summary()

# Use the encoder to calculate the embeddings for the data
X_train_autoenc = encoder.predict(X_train)
X_test_autoenc = encoder.predict(X_test)

# Calculate the embedding using PCA
pca = PCA(n_components=embedding_dim)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train three SVC models on the original data and the 2 embeddings
base_model = LinearSVC(C=1)
base_model.fit(X_train,y_train)

autoenc_model = LinearSVC(C=1)
autoenc_model.fit(X_train_autoenc,y_train)

pca_model = LinearSVC(C=1)
pca_model.fit(X_train_pca,y_train)

# Output performance info
print(f"The base SVM classifier scores {base_model.score(X_test, y_test)}.")
print(f"The autoencoder embedding SVM classifier scores {autoenc_model.score(X_test_autoenc, y_test)}.")
print(f"The PCA embedding SVM classifier scores {pca_model.score(X_test_pca, y_test)}.")