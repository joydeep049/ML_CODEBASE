"""### Stacked Auto-Encoders
Contains multiple encoders and decoders.
Used for encoding and decoding more complex data.
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import keras.layers as layers

(X_train,Y_train), (X_test, Y_test)= tf.keras.datasets.mnist.load_data()

X_train.shape

X_train= X_train/255.0

X_train= X_train.reshape(60000,-1)
N,D= X_train.shape
D

# Define the stacked Encoding and Decoding Layers using Functional API.

inputs= keras.Input(shape=(D,))

encoder1= layers.Dense(128,activation='relu' )(inputs)

encoder2= layers.Dense(64, activation = 'relu')(encoder1)

bottleneck_layer= layers.Dense(32, activation = 'relu')(encoder2)

decoder1= layers.Dense(64, activation='relu')(bottleneck_layer)

decoder2 =layers.Dense(128, activation ='relu')(decoder1)

decoder3= layers.Dense(D, activation = 'sigmoid')(decoder2)

autoencoder= keras.Model(inputs= inputs, outputs=decoder3)
encoder= keras.Model(inputs= inputs, outputs= bottleneck_layer)

autoencoder.compile(loss= 'binary_crossentropy',
              optimizer='adam')

autoencoder.fit(X_train,X_train, epochs= 20)
