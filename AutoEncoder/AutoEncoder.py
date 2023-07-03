### Functional API

It involves declaring each layer of a model differently.
The model trains as a whole, But we can see the outputs of the different functions seen in between.
Used widely for feature extraction as we can obtain the outputs of each subsequent layers for output visulization.
"""

inputs= keras.layers.Input(shape=(28,28,1))
conv1= keras.layers.Conv2D(64,activation= 'relu', kernel_size= (3,3), strides= (2,2))(inputs)
max_pool1= keras.layers.MaxPooling2D()(conv1)
conv2= keras.layers.Conv2D(32, activation= 'relu', kernel_size= (3,3), strides= (2,2))(max_pool1)
max_pool2= keras.layers.MaxPooling2D()(conv2)
flatten= keras.layers.Flatten()(max_pool2)

dense1= keras.layers.Dense(128, activation= 'relu')(flatten)
dense2= keras.layers.Dense(64, activation= 'relu')(dense1)
final_layer= keras.layers.Dense(10, activation ='relu')(dense2)

model= keras.Model(inputs= inputs, outputs= final_layer)

model.summary()
## Too much information loss in this convolutional neural network.

"""### A simple Autoencoder"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

# Generate data
def generate(a):
  angles= np.random.rand(a) * 0.1 * np.pi
  data= np.empty((a,3))
  data[:,0]= np.cos(angles)+  np.sin(angles) + 0.3 *np.random.randn(a)/2
  data[:,1]= np.cos (angles) * 0.3 + np.sin(angles)/2+ np.random.randn(a)
  data[:,2]= 0.1  * data[:,0] + 0.2 * data[:,1] + np.random.randn(a)

  return data

X_train = generate(100)
X_train = X_train - X_train.mean(axis= 0 , keepdims= 0 )

# plot the points
ax= plt.axes(projection='3d')
ax.scatter3D(X_train[:,0],X_train[:,1], X_train[:,2], cmap='red')

# Create an autoencoder model
encoder= keras.Sequential([
    keras.layers.Dense(2, input_shape= (3,))
])

decoder= keras.Sequential([
    keras.layers.Dense(3,input_shape=(2,))
])

autoencoder= keras.Sequential([encoder, decoder])

autoencoder.compile(loss='MSE', optimizer = 'SGD' )

autoencoder.fit(X_train, X_train,epochs = 200 )

codings= encoder.predict(X_train)

print("Actual Point", X_train[0])
print("Encoded Point", codings[0])

# Plot the dimensionally reduced points.
plt.title("Plot the dimensionally reduced points")
plt.scatter(codings[:,0], codings[:, 1])

encodings= decoder.predict(codings)

print("Actual Point", X_train[0])
print("Decoded Point",encodings[0] )

ax= plt.axes(projection='3d')
ax.scatter3D(X_train[:,0],X_train[:,1], X_train[:,2], cmap='red')

a= plt.axes(projection='3d')
a.scatter3D(encodings[:,0],encodings[:,1], encodings[:,2], cmap='red')

"""Hence , This is another method of dimensionality reduction.
