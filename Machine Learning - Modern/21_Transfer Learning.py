# -*- coding: utf-8 -*-
""" Machine Learning Exercise 21 - Transfer Learning - Emerson Ham
For this exercise, transfer learning is used to retrain the VGG-16 net to classify CIFAR10 images
"""

# ! pip install tensorflow-gpu==2.0

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# Find GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Load CIFAR10 Dataset and split into train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes=10
x_train.shape, y_train.shape
y_train.min(), y_train.max()
x_test.shape, y_test.shape

# Convert from class number output to binary class outputs
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train.shape, y_test.shape, y_train.min(), y_train.max()

# Create data augmentor since we have little data to work with
batch_size = 128
dataaugmentor = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
iter_train = dataaugmentor.flow(x_train, y_train, batch_size=batch_size)

# Load a VGG16 model with imagenet weights, without the fully connected layers, and with the adjusted input shape for our data
vgg_model = VGG16(include_top=False, input_shape=(32,32,3))

example_img = x_train[10]
example_img.shape

plt.imshow(example)
plt.show()

# Use the vgg_model you loaded to extract the features for the example image
# save the output to example_vgg

# YOUR CODE HERE
example_img = example_img.reshape((1, example.shape[0], example.shape[1], example.shape[2]))
example_vgg = vgg_model.predict(example_img)

# Create new fully connected layers for CIFAR10 classification
x = vgg_model.output
x = Flatten()(x)
x = Dense(128,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(64,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(32,activation="relu")(x)
predictions = Dense(10,activation="softmax")(x)

# Combine inputs and layers into the model
vgg_complete = Model(vgg_model.input, predictions)

# Set all of the original VGG16 layers to be non-trainable
for layer in vgg_model.layers:
    layer.trainable=False

vgg_complete.summary()

# Train the new fully connected layers
vgg_complete.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
steps = int(x_train.shape[0] / batch_size)
vgg_complete.fit_generator(iter_train, steps_per_epoch=steps, epochs=25, validation_data=(x_test, y_test))

# Report the accuracy on the test set
[loss, acc] = vgg_complete.evaluate(x_test,y_test)

assert acc > 0.55

print(f"If you were to guess just one class all the time, your accuracy would be 0.1. This model's accuracy is {acc:.3f}")
print("Overfitting is an issue with how small the dataset is(50k labels) and a similar number of terms. Data augmentation helps, but a larger dataset would certainly allow better accuracy.")