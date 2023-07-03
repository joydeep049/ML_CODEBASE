### A more advanced autoencoder on MNIST dataset.
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Create inputs
inputs= keras.layers.Input(shape=(784,))

def autoencoder():


  encoder= keras.layers.Dense(32, activation='relu')(inputs)

  decoder= keras.layers.Dense(784, activation = 'sigmoid')(encoder)

  return encoder, decoder

encoder_output, decoder_output  = autoencoder()

encoder_model= tf.keras.Model(inputs= inputs, outputs= encoder_output)

autoencoder_model= tf.keras.Model(inputs= inputs, outputs= decoder_output)

autoencoder_model.summary()

(X_train,Y_train), (X_test, Y_test)= tf.keras.datasets.mnist.load_data()

X_train= X_train/255.0
X_test= X_test/255.0

autoencoder_model.compile(loss= 'binary_crossentropy',
                          optimizer= 'adam')

X_train= X_train.reshape(60000,-1)

autoencoder_model.fit(X_train, X_train, epochs= 10)

encodings= encoder_model.predict(X_train)

X_train= X_train.reshape(60000,28,-1)

plt.matshow(X_train[0])

X_train= X_train.reshape(60000, -1)

decodings= autoencoder_model.predict(X_train)

decodings= decodings.reshape(60000,28,-1)

plt.matshow(decodings[0])
