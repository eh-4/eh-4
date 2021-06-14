# -*- coding: utf-8 -*-
""" Machine Learning Exercise 20 - GANs - Emerson Ham
In this exercise I build a GAN for mimicing the MNIST handwriting dataset
This version assumes tensorflow-gpu is installed. Otherwise it takes forever to run(70x longer)
"""

# ! pip install tensorflow-gpu==2.0

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dropout, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(0)

# Find GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Load data and convert from unsigned ints to floats and scale to 0 to 1
(x_train, _), (_, _) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_train = x_train.astype('float32') / 255.0

# Create the layers for the generator model
latent_dim = 256 # This is the dimension of the random noise we'll use for the generator
generator_layers = [ Dense(128*7*7, input_dim=latent_dim),
    LeakyReLU(alpha=0.2),
    Reshape((7,7,128)),
    Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2D(1,(7,7), activation='sigmoid', padding='same')]

generator = Sequential(generator_layers)
generator.summary()

# Create the layers for the discriminator model
discriminator_layers = [ Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(28,28,1)),
    LeakyReLU(alpha=0.2),
    Dropout(0.4),
    Conv2D(64, (3,3), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Dropout(0.4),
    Flatten(),
    Dense(1, activation='sigmoid')]

discriminator = Sequential(discriminator_layers)
discriminator.summary()

# Assemble the gan
discriminator.trainable = False
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.summary()

# Compile the gan
opt = Adam(lr=0.0002, beta_1=0.5)
gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Compile the discriminator
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Hyperparameters
batch_size = 256
epochs = 100
n_batches = int(x_train.shape[0] / batch_size)
half_batch = int(batch_size / 2)

# Adversarial ground truths
g_labels = np.ones((batch_size, 1))
d_labels = np.vstack((np.ones((half_batch, 1)), np.zeros((half_batch, 1))))

# Train
for epoch in range(epochs):
  for batch in range(n_batches):
    print(".", end = '')
    # Create labels for real and generated halves
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    real_imgs = x_train[idx]
    noise = np.random.normal(0, 1, size = (half_batch, latent_dim))
    gen_imgs = generator.predict(noise)

    # Train the discriminator
    imgs = np.vstack((real_imgs, gen_imgs))
    _, d_loss = discriminator.train_on_batch(imgs, d_labels)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    _, g_loss = gan.train_on_batch(noise, g_labels)
  
  # Output
  if epoch % 5 == 0:
    r, c = 2,5
    noise = np.random.normal(0, 1, (r*c, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = np.squeeze(gen_imgs)

    fig, axs = plt.subplots(r, c)
    
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.suptitle(f"Epoch: {epoch+1} d_loss: {d_loss} g_loss: {g_loss}")
    plt.show()
  print('')

# Generate sample digit
noise = np.random.normal(0, 1, (1, latent_dim))
gen_img = generator.predict(noise)[0]
gen_img = gen_img.squeeze()
plt.imshow(gen_img, cmap='gray')
plt.show()