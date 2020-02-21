""" This module implements variational autoencoder models in tensorflow / keras.
"""
import tensorflow as tf
import numpy as np

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, kernel_size):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = make_vae_inference_net(latent_dim)
        self.generative_net = make_vae_generative_net(latent_dim, kernel_size)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def make_vae_generative_net(latent_dim, kernel_size):
    model = tf.keras.Sequential(name='Decoder')
    
    model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Reshape(target_shape=(16, 16, 32)))
    assert model.output_shape == (None, 16, 16, 32)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 32, 32, 128)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 64, 64, 128)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 128, 128, 128)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 256, 256, 128)
    
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=(1, 1), padding="SAME"))
    return model



def make_vae_generative_net(latent_dim, kernel_size):
    model = tf.keras.Sequential(name='Decoder')
    
    model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Reshape(target_shape=(16, 16, 32)))
    assert model.output_shape == (None, 16, 16, 32)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))


    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 32, 32, 128)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 64, 64, 128)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 128, 128, 128)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 256, 256, 128)
    
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=(1, 1), padding="SAME"))
    return model

def decoder_inception_net(latent_dim):
    model = tf.keras.Sequential(name='Decoder')
    
    model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
    model.add(tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Reshape(target_shape=(16, 16, 32)))
    assert model.output_shape == (None, 16, 16, 32)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add()
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 32, 32, 64)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 64, 64, 64)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 128, 128, 64)
    
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    assert model.output_shape == (None, 256, 256, 64)
    
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=1, strides=(1, 1), padding="SAME"))
    return model


# function for creating a naive inception block
def inception_module(layer_in, f1, f2, f3):
    """ An implementation of the Inception module.
    """
    # 1x1 conv
    conv1 = tf.keras.layers.Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = tf.keras.layers.Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
    # 5x5 conv
    conv5 = tf.keras.layers.Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
    # 3x3 max pooling
    pool = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = tf.keras.layers.Concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

def make_vae_inference_net(latent_dim):
    model = tf.keras.Sequential(name='Encoder')
    
    model.add(tf.keras.layers.InputLayer(input_shape=(256, 256, 1)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding="SAME", strides=(2, 2), activation='relu'))
    assert model.output_shape == (None, 128, 128, 256)
    
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="SAME", strides=(2, 2), activation='relu'))
    assert model.output_shape == (None, 64, 64, 128)
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding="SAME", strides=(2, 2), activation='relu'))
    assert model.output_shape == (None, 32, 32, 64)
    
    model.add(tf.keras.layers.Flatten())
    assert model.output_shape == (None, 32 * 32 * 64)
    
    model.add(tf.keras.layers.Dense(latent_dim + latent_dim))
    return model


@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.keras.losses.MSE(x, x_logit)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
    logpz = log_normal_pdf(z, tf.constant(0.), tf.constant(0.))
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)