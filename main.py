import os
import numpy as np
import keras
from conditional_dcgan import ConditionalDCGAN
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical


def normalize(X):
    return (X - 127.5)/127.5


def denormalize(X):
    return (X + 1.0)*127.5


def train(latent_dim, height, width, channels, num_class):
    (X_train, Y_train), (_, _) = keras.datasets.mnist.load_data()
    Y_train = to_categorical(Y_train, num_class)
    X_train = X_train.reshape(
        (X_train.shape[0],) + (height, width, channels)).astype('float32')
    X_train = normalize(X_train)
    epochs = 50
    batch_size = 128
    iterations = X_train.shape[0]//batch_size
    dcgan = ConditionalDCGAN(latent_dim, height, width, channels, num_class)
    for epoch in range(epochs):
        for iteration in range(iterations):
            real_images = X_train[iteration *
                                  batch_size:(iteration+1)*batch_size]
            conditions = Y_train[iteration*batch_size:(iteration+1)*batch_size]
            d_loss, g_loss = dcgan.train(real_images, conditions, batch_size)
            if (iteration + 1) % 1 == 0:
                print("epochs:{}/{}, iterations:{}/{}".format(epoch, epochs, iteration, iterations))
                print('discriminator loss:', d_loss)
                print('generator loss:', g_loss)
                print()
                with open('loss.txt', 'a') as f:
                    f.write(str(d_loss) + ',' + str(g_loss) + '\r')
        if (epoch + 1) % 1 == 0:
            dcgan.save_weights('gan' + '_epoch' + str(epoch + 1) + '.h5')
            random_latent_vectors = np.random.normal(
                size=(batch_size, latent_dim))
            generated_images = dcgan.predict(random_latent_vectors, conditions)
            for i, generated_image in enumerate(generated_images):
                img = denormalize(generated_image)
                img = image.array_to_img(img, scale=False)
                condition = np.argmax(conditions[i])
                filename = os.path.join('generated', str(
                    epoch) + '_' + str(condition) + '.png')
                print("saved image:", filename)
                img.save(filename)
        print('epoch' + str(epoch) + ' end')
        print()


def predict(latent_dim, height, width, channels, num_class):
    dcgan = ConditionalDCGAN(latent_dim, height, width, channels, num_class)
    dcgan.load_weights('gan_epoch50.h5')
    for num in range(num_class):
        for id in range(10):
            random_latent_vectors = np.random.normal(size=(1, latent_dim))
            condition = np.zeros((1, num_class), dtype=np.float32)
            condition[0, num] = 1
            generated_images = dcgan.predict(random_latent_vectors, condition)
            img = image.array_to_img(denormalize(
                generated_images[0]), scale=False)
            img.save(os.path.join('generated', str(
                num) + '_' + str(id) + '.png'))


if __name__ == '__main__':
    latent_dim = 100
    height = 28
    width = 28
    channels = 1
    num_class = 10
    train(latent_dim, height, width, channels, num_class)
    predict(latent_dim, height, width, channels, num_class)
