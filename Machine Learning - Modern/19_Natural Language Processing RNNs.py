# -*- coding: utf-8 -*-
""" Machine Learning Exerciese 19 - RNNs - Emerson Ham
This exercise covers the use of RNNs, including GRU, and LSTMs for the classification of news article sentiment
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Embedding, Dense
from tensorflow.keras.datasets import imdb

# Load IMDB movie review dataset from keras
maxlen = 100 # Only use sentences up to this many words
n = 20000 # Only use the most frequent n words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n)

x_train.shape
x_test.shape

for i in range(10):
    print(f"Element {i} has a length of {len(x_train[i])}")

x_train[0][:10]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

x_train.shape, x_test.shape

print(f"All values of the targets are integers with the following max and min values")
print(f"{y_train.max()}, {y_train.min()}")

# Define simple RNN
simple_layers = [ Embedding(input_dim=n, output_dim=64),
    SimpleRNN(units=64,dropout=0.1, recurrent_dropout=0.1),
    Dense(1, activation='sigmoid')]

my_simple = Sequential(simple_layers)
my_simple.summary()

my_simple.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
my_simple.fit(x_train, y_train, batch_size=32, epochs=1)

# Define GRU Layers
gru_layers = [ Embedding(input_dim=n, output_dim=64),
    GRU(units=64,dropout=0.1, recurrent_dropout=0.1),
    Dense(1, activation='sigmoid')]

my_gru = Sequential(gru_layers)
my_gru.summary()

my_gru.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
my_gru.fit(x_train, y_train, batch_size=32, epochs=1)

# Define LSTM Layers
lstm_layers = [ Embedding(input_dim=n, output_dim=64),
    LSTM(units=64,dropout=0.1, recurrent_dropout=0.1),
    Dense(1, activation='sigmoid')]

my_lstm = Sequential(lstm_layers)
my_lstm.summary()

my_lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
my_lstm.fit(x_train, y_train, batch_size=32, epochs=1)

# Evaluate accuracy of all 3 models
[my_simple_loss, my_simple_acc] = my_simple.evaluate(x_test,y_test)
[my_gru_loss, my_gru_acc] = my_gru.evaluate(x_test,y_test)
[my_lstm_loss, my_lstm_acc] = my_lstm.evaluate(x_test,y_test)

print(f"Your simple model achieved an accuracy of {my_simple_acc:.2}.")
print(f"Your GRU model achieved an accuracy of {my_gru_acc:.2}.")
print(f"Your LSTM model achieved an accuracy of {my_lstm_acc:.2}.")

assert my_simple_acc > 0.4
assert my_gru_acc > 0.6
assert my_lstm_acc > 0.7