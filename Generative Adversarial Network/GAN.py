from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Flatten, Dense, Lambda, LeakyReLU, Dropout, BatchNormalization, GaussianNoise
from numpy import prod, add, array, zeros, ones, log
from numpy.random import normal, randint
import tensorflow as tf


def Data():

    data_file = open("mnist_data.txt", 'r', encoding='utf-8')

    data = []

    for line in data_file:
        entry = array(line.split(), dtype=float)
        data.append(entry)

    x = zeros((len(data), 784))

    for i, entry in enumerate(data):
        x[i] = entry

    # normalize inputs
    x = x / 127.5 - 1

    return x

def Generator(latent, img_shape, h0=256, h1=512, h2=1024 , d = 0.4, m=0.8, alpha=0.2):
    """Returns sequential generator model, takes in latent dim, image shape = (int, int), hidden sizes h0, h1, h2
       dropout(d), momentum(m) for batch normalization, and alpha for leaky relu"""

    # input

    z = Input(shape=(latent,))

    # first hidden layer

    h = z
    h = Dense(h0)(h)
    h = BatchNormalization(momentum=m)(h)
    h = LeakyReLU(alpha)(h)
    h = Dropout(d)(h)

    # second

    h = Dense(h1)(h)
    h = BatchNormalization(momentum=m)(h)
    h = LeakyReLU(alpha)(h)
    h = Dropout(d)(h)

    # third

    h = Dense(h2)(h)
    h = BatchNormalization(momentum=m)(h)
    h = LeakyReLU(alpha)(h)
    h = Dropout(d)(h)

    # final hidden layer, dependent on image shape

    h = Dense(prod(img_shape), activation='tanh')(h)

    # reshape

    h = Reshape(img_shape)(h)

    model = Model(z, h)
    modelt = Model(z, h)
    modelt.trainable = False

    return model, modelt

def Discriminator(img_shape, h0=512, h1=256, d=0.4, alpha=0.2):
    """ Feed image shape = (int, int), hidden sizes = [int, int], dropout, momentum for batch normal, and alpha for
        leaky relu """

    # input layer

    x = Input(shape=img_shape)

    h = x

    # flatten input
    h = Flatten(input_shape=img_shape)(h)

    # first hidden layer

    h = Dense(h0)(h)
    h = LeakyReLU(alpha)(h)
    h = Dropout(d)(h)

    # second

    h = Dense(h1)(h)
    h = LeakyReLU(alpha)(h)
    h = Dropout(alpha)(h)

    # output layer

    h = Dense(1, activation='sigmoid')(h)

    # model

    model = Model(x, h)
    modelt = Model(x, h)
    modelt.trainable = False

    return model, modelt




latent, img_shape, input_size, k, = 256, (28, 28), 784, 2

x = Data()

samples = x.shape[0]
epochs = 100
batch_size = 128
batches = int(samples / batch_size)

Gx = zeros((batch_size, latent))
Gy = zeros((batch_size, 1))

Dx0 = zeros((batch_size, img_shape[0], img_shape[1]))
Dx1 = zeros((batch_size, img_shape[0], img_shape[1]))

Dy0 = zeros((batch_size, 1))
Dy1 = ones((batch_size, 1))

noise = normal(0, 1, (samples, latent))

G, Gt = Generator(latent, img_shape)
D, Dt = Discriminator(img_shape)

Adam = tf.train.AdamOptimizer

optimizer = Adam(1e-4, 0.2)

z = Input(shape=(latent,))
x = Input(shape=(input_size,))

sess = K.get_session()


# for each epoch
for epoch in range(epochs):

    #for batch in batches
    for batch in range(batches):

        # train discriminator for k batches
        for step in range(k):

            # for each sample in the batch
            for sample in range(batch_size):

                i = randint(0, samples)

                Gx[sample] = noise[i]

                i = randint(0, samples)

                Dx1[sample] = x[i].reshape((28, 28))

            Dy0 = D.predict(G.predict(Gx))
            Dy1 = D.predict(Dx1)

            dloss = 0.0

            # accumulate loss for each sample
            for sample in range(batch_size):

                dloss += log(1 - Dy0[sample]) + 0.1 * log(1 - Dy1[sample])

            # take average

            dloss /= batch_size

            # get loss weighted gradients (delta D)

            deltaD = optimizer.compute_gradients(dloss, D.trainable_weights)

            # get updates

            upD = optimizer.apply_gradients(deltaD)

            # run ression

            result  = sess.run([upD, dloss])

        # for each sample in the batch
        for sample in range(batch_size):

            i = randint(0, samples)

            Gx[sample] = noise[i]








