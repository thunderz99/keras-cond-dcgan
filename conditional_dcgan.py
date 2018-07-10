import numpy as np
import keras
from keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2D, MaxPool2D, Flatten, concatenate, multiply
from keras.models import Model

class Generator(object):
    def __init__(self, latent_dim, condition_dim):
        # latent vector input
        generator_input1 = Input(shape=(latent_dim,))
        # condition input
        generator_input2 = Input(shape=(condition_dim,))
        # concat 2 inputs
        generator_input = concatenate([generator_input1, generator_input2])
        x = Dense(1024)(generator_input)
        x = Activation('tanh')(x)
        x = Dense(128*7*7)(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Reshape((7, 7, 128))(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, 5, padding='same')(x)
        x = Activation('tanh')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(1, 5, padding='same')(x)
        x = Activation('tanh')(x)
        # pass condition input to output so we can give it to discriminator
        self.generator = Model(inputs=[generator_input1, generator_input2], outputs=[x, generator_input2])

    def get_model(self):
        return self.generator


class Discriminator(object):
    def __init__(self, height, width, channels, condition_dim):
        # real or fake image
        discriminator_input1 = Input(shape=(height, width, channels))
        # condition input from generator
        discriminator_input2 = Input(shape=(condition_dim,))
        # expand dimension from (batch, channel) to (batch, height, width, channel)
        di2 = Reshape((1, 1, condition_dim))(discriminator_input2)
        # expand height and width from (1, 1) to (height, width)
        di2 = UpSampling2D((height, width))(di2)
        # concat 2 inputs
        discriminator_input = concatenate([discriminator_input1, di2])
        x = Conv2D(64, 5, padding='same')(discriminator_input)
        x = Activation('tanh')(x)
        x = MaxPool2D()(x)
        x = Conv2D(128, 5)(x)
        x = Activation('tanh')(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('tanh')(x)
        x = Dense(1, activation='sigmoid')(x)
        self.discriminator = Model(inputs=[discriminator_input1, discriminator_input2], outputs=x)

    def get_model(self):
        return self.discriminator


class ConditionalDCGAN(object):
    def __init__(self, latent_dim, height, width, channels, condition_dim):
        # set generator
        self._latent_dim = latent_dim
        g = Generator(latent_dim, condition_dim)
        self._generator = g.get_model()
        # set discriminator
        d = Discriminator(height, width, channels, condition_dim)
        self._discriminator = d.get_model()
        # compile discriminator
        discriminator_optimizer = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self._discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
        # disable training when combined with generator
        self._discriminator.trainable = False
        # set DCGAN
        dcgan_input1 = Input(shape=(latent_dim,))
        dcgan_input2 = Input(shape=(condition_dim,))
        dcgan_output = self._discriminator(self._generator([dcgan_input1, dcgan_input2]))
        self._dcgan = Model([dcgan_input1, dcgan_input2], dcgan_output)
        # compile DCGAN
        dcgan_optimizer = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self._dcgan.compile(optimizer=dcgan_optimizer, loss='binary_crossentropy')

    def train(self, real_images, conditions, batch_size):
        # Train discriminator so it can detect fake
        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        generated_images = self._generator.predict([random_latent_vectors, conditions])
        labels = np.ones((batch_size, 1))
        labels += 0.05 * np.random.random(labels.shape)
        d_loss1 = self._discriminator.train_on_batch(generated_images, labels)
        # Train discriminator so it can detect real
        labels = np.zeros((batch_size, 1))
        labels += 0.05 * np.random.random(labels.shape)
        d_loss2 = self._discriminator.train_on_batch([real_images, conditions], labels)
        d_loss = (d_loss1 + d_loss2)/2.0
        # Train generator so it can fool discriminator
        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        misleading_targets = np.zeros((batch_size, 1))
        g_loss = self._dcgan.train_on_batch([random_latent_vectors, conditions], misleading_targets)
        return d_loss, g_loss

    def predict(self, latent_vector, condition):
        # return only image (remember generator returns condition too)
        return self._generator.predict([latent_vector, condition])[0]

    def load_weights(self, file_path, by_name=False):
        self._dcgan.load_weights(file_path, by_name)

    def save_weights(self, file_path, overwrite=True):
        self._dcgan.save_weights(file_path, overwrite)
