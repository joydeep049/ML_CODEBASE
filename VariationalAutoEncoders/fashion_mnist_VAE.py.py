# -*- coding: utf-8 -*- 

# Code

## Essential imports
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from IPython import display

"""##Parameters"""

BATCH_SIZE= 128
LATENT_DIM= 2

"""## Prepare the Dataset

You will just be using the `train` split of the MNIST dataset in this notebook. We've prepared a few helper functions below to help in downloading and preparing the dataset:

* `map_image()` - normalizes and creates a tensor from the image, returning only the image. This will be used for the unsupervised learning in the autoencoder.

* `get_dataset()` - loads MNIST from Tensorflow Datasets, fetching the `train` split by default, then prepares it using the mapping function. If `is_validation` is set to `True`, then it will get the `test` split instead. Training sets will also be shuffled.
"""

def map_image(image,label):
  image= tf.cast(image, tf.float32)
  image= image/255.0
  image = tf.reshape(image, (28,28,1))

  return image

def get_dataset(map_fn, is_validation = False):
  if is_validation:
    split_name= "test"
  else:
    split_name= "train"
  dataset = tfds.load('fashion_mnist', as_supervised= True, split= split_name)
  dataset= dataset.map(map_fn)

  if is_validation:
    dataset= dataset.batch(BATCH_SIZE)
  else :
    dataset= dataset.shuffle(1024).batch(BATCH_SIZE)

  return dataset

train_dataset= get_dataset(map_image)

"""## Build the Model

You will now be building your VAE model. The main parts are shown in the figure below:


Like the autoencoder, the VAE also has an encoder-decoder architecture with the main difference being the grey box in the middle which stands for the latent representation. In this layer, the model mixes a random sample and combines it with the outputs of the encoder. This mechanism makes it useful for generating new content. Let's build these parts one-by-one in the next sections.

### Sampling Class

First, you will build the `Sampling` class. This will be a custom Keras layer that will provide the Gaussian noise input along with the mean (mu) and standard deviation (sigma) of the encoder's output. In practice, the output of this layer is given by the equation:

$$z = \mu + e^{0.5\sigma} * \epsilon  $$

where $\mu$ = mean, $\sigma$ = standard deviation, and $\epsilon$ = random sample
"""

class Sampling(tf.keras.layers.Layer):
  def call(self,inputs):

    myu, sigma= inputs

    batch= tf.shape(myu)[0]
    dim= tf.shape(myu)[1]

    epsilon = tf.keras.backend.random_normal(shape=(batch,dim))

    return (myu + tf.exp(0.5 * sigma) * epsilon)

"""### Encoder

Next, you will build the encoder part of the network. You will follow the architecture shown in class which looks like this. Note that aside from mu and sigma, you will also output the shape of features before flattening it. This will be useful when reconstructing the image later in the decoder.

*Note:* You might encounter issues with using batch normalization with smaller batches, and sometimes the advice is given to avoid using batch normalization when training VAEs in particular. Feel free to experiment with adding or removing it from this notebook to explore the effects.


"""

def encoder_layers(inputs, latent_dim):
  x= tf.keras.layers.Conv2D(32, 3, (2,2), padding='same', activation= 'relu', name= 'encode_conv1')(inputs)
  x= tf.keras.layers.BatchNormalization()(x)
  x= tf.keras.layers.Conv2D(64, 3, (2,2), padding='same', activation= 'relu', name= 'encode_conv2')(x)

  batch_2=tf.keras.layers.BatchNormalization()(x)

  x= tf.keras.layers.Flatten(name='encode_flatten')(batch_2)

  x= tf.keras.layers.Dense(20, activation = 'relu', name='encode_dense')(x)
  x= tf.keras.layers.BatchNormalization()(x)

  mu= tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)
  sigma= tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

  return mu, sigma, batch_2.shape

def encoder_model(latent_dim, input_shape):
  inputs= tf.keras.Input(shape= input_shape)

  mu, sigma, conv_shape= encoder_layers(inputs, latent_dim=LATENT_DIM)

  z= Sampling()((mu, sigma))

  model= tf.keras.Model(inputs=inputs, outputs= [mu,sigma,z] )

  return model, conv_shape

"""UpSampling2D is like the opposite of pooling where it repeats rows and columns of the input. Conv2DTranspose performs up-sampling and convolution.

"""

def decoder_layers(inputs, conv_shape):

  units= conv_shape[1]* conv_shape[2] *conv_shape[3]
  x= keras.layers.Dense(units, activation='relu', name='decode_dense1')(inputs)
  x= keras.layers.BatchNormalization()(x)

  x= tf.keras.layers.Reshape((conv_shape[1],conv_shape[2],conv_shape[3]), name='decode_reshape')(x)

  x= tf.keras.layers.Conv2DTranspose(64, 3,2,padding='same', activation='relu',name='decode_conv1')(x)
  x= keras.layers.BatchNormalization()(x)
  x= keras.layers.Conv2DTranspose(32, 3,2,padding='same', activation='relu', name= 'decode_conv2')(x)
  x= keras.layers.BatchNormalization()(x)

  # Final Decoder Layer
  x= keras.layers.Conv2DTranspose(1,3,1,padding='same', activation='sigmoid',name='decode_final')(x)

  return x

def decoder_model(latent_dim, conv_shape):

  inputs= keras.Input(shape=(latent_dim,))

  outputs= decoder_layers(inputs, conv_shape)

  model= tf.keras.Model(inputs= inputs,outputs= outputs)

  return model

def kl_reconstruction_loss(mu, sigma):
  """
  Returns the Kullback-leibler Divergence .
  It is a probabilistic metric that helps us to find the excess surprise when a probability distribution Q is used instead of the original Distribution P.
  It also is a measure of the similarity between the two Probability Distributions.
  XXXXXXX
  mu- Mean
  sigma- Standard Deviation
  XXXXXXX
  Return the KLD Loss.
  """
  kl_loss= 1+ sigma- tf.square(mu)- tf.math.exp(sigma)
  kl_loss= tf.reduce_mean(kl_loss) * -0.5
  return kl_loss

def vae_model(encoder, decoder, input_shape):
  """
  Returns the Final Variational Autoencoder Model.
  -------
  Encoder --> The Encoder Model
  Decoder --> The Decoder Model
  input_shape --> Shape of the dataset
  ------
  Returns the VAE Model.
  """
  inputs= tf.keras.Input(shape=input_shape)

  mu,sigma,z= encoder(inputs)

  reconstructed= decoder(z)

  vae_model= keras.Model(inputs= inputs, outputs= reconstructed)

  loss= kl_reconstruction_loss(mu,sigma)
  vae_model.add_loss(loss)

  return vae_model

def get_models(input_shape, latent_dim):
  """Returns the encoder, decoder, and vae models"""
  encoder, conv_shape = encoder_model(latent_dim=latent_dim, input_shape=input_shape)
  decoder = decoder_model(latent_dim=latent_dim, conv_shape=conv_shape)
  vae = vae_model(encoder, decoder, input_shape=input_shape)
  return encoder, decoder, vae

encoder, decoder,vae= get_models(input_shape=(28,28,1), latent_dim= LATENT_DIM)

encoder.summary()

decoder.summary()

"""## TRAIN THE MODEL"""

optimizer= tf.keras.optimizers.Adam()
bce_loss= tf.keras.losses.BinaryCrossentropy()
loss_metric= tf.keras.metrics.Mean()

"""You will want to see the progress of the image generation at each epoch. For that, you can use the helper function below. This will generate 16 images in a 4x4 grid."""

def generate_and_save_images(model, epoch, step, test_input):
  """
  Helper function to generate 16 images.
  --------
  model--> Decoder Model
  epoch--> Current Epoch Number
  step-->  Current step number
  test_input--> Random Tensor of shape (16, LATENT_DIM)
  """
  predictions= model.predict(test_input)

  plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
    plt.subplot(4,4,i+1)
    plt.imshow(predictions[i,:,:,0], cmap='gray')
    plt.axis('off')

  plt.suptitle('epoch->{}, step->{}'.format(epoch,step))
  plt.savefig('image_at_epoch{}_step_{}'.format(epoch,step))
  plt.show()

# Training Loop

random_tensor_for_generation = tf.random.normal(shape=[16,LATENT_DIM])

epochs= 1000

for i in range(epochs):
  print("Start of epoch {}".format(i+1))

  for step, x_train_batch in enumerate(train_dataset):
    with tf.GradientTape() as tape:
      # Feed a batch to the VAE Model
      reconstructed= vae(x_train_batch)

      # Calculate Reconstruction error
      flattened_inputs= tf.reshape(x_train_batch , shape=[-1])
      flattened_outputs= tf.reshape(reconstructed , shape=[-1])
      loss= bce_loss(flattened_inputs, flattened_outputs)* 3072

      loss+=sum(vae.losses)

      # Calculating gradients and Applying gradients to the weights by Backpropagating through the Computational Graph of the neural network.
      gradients= tape.gradient(loss, vae.trainable_weights)
      optimizer.apply_gradients(zip(gradients,vae.trainable_weights))

      loss_metric(loss)

      # Display outputs every 100 steps:
      if step%100== 0:
        display.clear_output(wait=False)
        generate_and_save_images(decoder,i+1,step,random_tensor_for_generation)
        print('Epoch: %s step: %s mean loss = %s' % (i+1, step, loss_metric.result().numpy()))