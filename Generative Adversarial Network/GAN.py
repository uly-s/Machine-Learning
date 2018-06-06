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

def Generator(latent, img_shape, h0=256, h1=512, h2=1024, d=0.4, m=0.8, alpha=0.2):
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

    return model

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
    h = Dropout(d)(h)

    # output layer

    h = Dense(1, activation='sigmoid')(h)

    # model

    model = Model(x, h)

    return model

latent, img_shape, input_size, k, = 256, (28, 28), 784, 1

X = Data()  # get mnist text file

num_samples = X.shape[0]
epochs = 100
batch_size = 128
batches = int(num_samples / batch_size)

noise = zeros((batch_size, latent))  # noise to feed into generator
generated = zeros((batch_size, 1))   # generated images

real = zeros((batch_size, img_shape[0], img_shape[1]))  # real images for discriminator

# create noise

noise_prior = normal(0, 1, (num_samples, latent))

# create gen and disc

G = Generator(latent, img_shape)
D = Discriminator(img_shape)

# set inputs

z = Input(shape=(latent,))      # noise input
x = Input(shape=img_shape)      # image input

# set models

Gz = G(z)       # noise into generator
Dz = D(Gz)      # noise into gen into disc
Dx = D(x)       # real into disc

# create optimizers

Adam = tf.train.AdamOptimizer       # need two optimizers because the internal variables
optG = Adam(1e-4, 0.2)              # of adam will be different for each model
optD = Adam(1e-4, 0.2)

# create loss functions

gloss = - K.mean(K.log(Dz))     # research paper hs (1 - Dz) but loss goes to 0 (suggested on github)
dloss = - K.mean(K.log(Dx) + K.log(1 - Dz))

# set operations for updating, compute gradients between loss and weights, get updates
minG = optG.apply_gradients(optG.compute_gradients(gloss, G.trainable_weights))
minD = optD.apply_gradients(optD.compute_gradients(dloss, D.trainable_weights))

# init
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

# training session
with tf.Session() as sess:
    sess.run(init_global)
    sess.run(init_local)

    # for each epoch
    for epoch in range(epochs):

        #for batch in batches
        for batch in range(batches):

            # train discriminator for k batches
            for step in range(k):

                # for each sample in the batch
                for sample in range(batch_size):

                    i = randint(0, num_samples)     # get a random sample from prior noise ( interpreted from paper)

                    noise[sample] = noise_prior[i]

                    i = randint(0, num_samples)     # get a random image

                    real[sample] = X[i].reshape((28, 28))

                # run ression

                _, batch_loss_D = sess.run([minD, dloss], feed_dict={z:noise, x:real})  # train on real and fake

            # for each sample in the batch
            for sample in range(batch_size):

                i = randint(0, num_samples)     # sample noise prior

                noise[sample] = noise_prior[i]

            _, batch_loss_G = sess.run([minG, gloss], feed_dict={z:noise})      # run minimize on noise batch

            print("D: " + str(batch_loss_D) + ", G: " + str(batch_loss_G))










