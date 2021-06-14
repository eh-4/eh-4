# -*- coding: utf-8 -*-
"""17 - DL - CNN.ipynb
This exercise is an introduction to CNNs for classification applied to the MNIST dataset
"""

import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10, mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(0)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
s1 = x_train.shape
s2 = x_test.shape
print(f"The mnist data was loaded with {s1[0]} training samples and {s2[0]} testing samples. Each sample is a {s1[1]} x {s1[2]} pixel image.")

example_img = x_train[0]
example_img
plt.imshow(example_img, cmap="gray", vmin=0, vmax=255)

def calculate_conv_shape(X, K, padding=0, stride=1):
    # Calculates the shape of the output of a convolution 
    # Inputs:
    #   X (np.array): The input matrix
    #   K (np.array): The filter matrix
    #   padding (int, optional): Defaults to 0. The padding dimension
    #   stride (int, optional): Defaults to 1. The stride of the convolution
    # Returns:
    #   tuple: The shape of the convolution output (height followed by width)
    
    Xw = X.shape[1]
    Xh = X.shape[0]
    Kw = K.shape[1]
    Kh = K.shape[0]

    yw = int((Xw-Kw+2*padding)/stride + 1)
    yh = int((Xh-Kh+2*padding)/stride + 1)
    
    return (yh,yw)

# Manualy create a very basic convolution for sharpening an image
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]])

calculate_conv_shape(example_img, sharpen, padding=1)

ans = calculate_conv_shape(example_img, sharpen, padding=1)
ans = calculate_conv_shape(example_img, sharpen, padding=0, stride=2)

# Apply the sharpen filter to the example_img and save the output to sharpened_image
print(example_img.shape)
print(sharpen.shape)
sharpened_image = scipy.ndimage.convolve(example_img, sharpen)

plt.imshow(sharpened_image, cmap="gray", vmin=0, vmax=255)
plt.show()

# Apply a filter of your choice (I chose a blur filter) and save the output image to filtered_image
my_filter = np.array([
    [1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]
])
filtered_image = scipy.ndimage.convolve(sharpened_image, my_filter)

plt.imshow(filtered_image, cmap="gray", vmin=0, vmax=255)
plt.show()

# Create a simple FFNN model for clasifying the images
simple_layers = [ Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(10, activation='softmax')]
simple_model = keras.Sequential(simple_layers)
simple_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the data and assess performance
simple_model.fit(x_train / 256., y_train, epochs=5)
simple_scores = simple_model.evaluate(x_test / 256., y_test)
print(simple_scores)
assert simple_scores[1] > 0.3

print(f"\nThe simple model achieves an accuracy of {simple_scores[1]*100:.2f}% on the test data.")

# Create a CNN model to compare with the FFNN
cnn_layers = [
    Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)),
    MaxPool2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')]
cnn_model = keras.Sequential(simple_layers)
cnn_model.compile(optimizer="adam",  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the CNN model and test its performance
cnn_model.fit(x_train.reshape(-1, 28, 28 ,1), y_train, epochs=5)
cnn_scores = cnn_model.evaluate(x_test.reshape(-1, 28, 28 ,1), y_test)
assert cnn_scores[1] > 0.9
print(f"\nThe CNN model achieves an accuracy of {cnn_scores[1]*100:.2f}% on the test data.")

# Compare output of the two models
# Change this value to test out some number images and see how the models perform
i = 245
new_example_img = x_test[i].astype(np.float32)

simple_new_example_img = new_example_img.reshape(-1, 28, 28)
cnn_new_example_img=new_example_img.reshape(-1, 28, 28, 1)

simple_predict = simple_model.predict(simple_new_example_img).argmax()
cnn_predict = cnn_model.predict(cnn_new_example_img).argmax()
target = y_test[i].astype(np.float32)

plt.imshow(new_example_img, cmap="gray", vmin=0, vmax=255)
print(f"The simple model predicts this image is a {simple_predict} and the CNN predicts it is a {cnn_predict}.")
print(f"The thing is a {target}")