"""### Convolutional Auto-Encoder"""

import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(X_train,Y_train),(X_test,Y_test)= keras.datasets.fashion_mnist.load_data()

X_train.shape

inputs= keras.Input(shape=(28,28,1))

def encoder(inputs):

  encoder_1= layers.Conv2D(32,kernel_size= (3,3),activation='relu',padding= 'same')(inputs) ## 28x28x32
  max_pool_1= layers.MaxPooling2D((2,2))(encoder_1)                                         ## 14x14x32

  encoder_2= layers.Conv2D(64, kernel_size=(3,3),padding='same', activation='relu')(max_pool_1) ## 14x14x64
  max_pool_2= layers.MaxPooling2D((2,2))(encoder_2)                                             ## 7x7x764

  return max_pool_2

def bottleneck(max_pool_2):
  bottleneck= layers.Conv2D(128, kernel_size=(3,3),activation='relu', padding='same')(max_pool_2) ## 7x7x128

  encoder_visualization = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(bottleneck) ## 7x7x1
  return bottleneck, encoder_visualization

def decoder(bottleneck):
  decoder_1= layers.Conv2D(64, activation='relu', padding='same', kernel_size=(3,3))(bottleneck) ## 7x7x64
  upsample_1= layers.UpSampling2D((2,2))(decoder_1)                                              ##14x14x64

  decoder_2= layers.Conv2D(32,activation='relu', padding='same', kernel_size=(3,3))(upsample_1)  ## 14x14x32
  upsample_2= layers.UpSampling2D((2,2))(decoder_2)                                              ## 28x28x32

  decoder= layers.Conv2D(1,activation='relu', padding='same', kernel_size=(3,3))(upsample_2)     ##28x28x1


  return decoder

encoder_output= encoder(inputs)

bottleneck_output,encoder_visualization= bottleneck(encoder_output)

decoder_output= decoder(bottleneck_output)

autoencoder= keras.Model(inputs= inputs, outputs= decoder_output)

encoder_model= keras.Model(inputs=inputs, outputs= encoder_visualization)

autoencoder.compile(loss='binary_crossentropy',
                    optimizer='adam')

X_train = X_train.reshape(60000,28,28,-1)

X_train= X_train/255.0

autoencoder.summary()

# autoencoder.fit(X_train, X_train, epochs=10)