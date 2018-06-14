from keras.backend import square, exp, log, shape, random_normal
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from numpy import array, zeros
from numpy.random import randint
from tensorflow import reduce_mean as mean, reduce_sum as sum
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
    x = x / 255

    return x

X = Data()

epochs, batch_size, samples = 100, 64, X.shape[0]
batches = int(samples / batch_size)

input, latent, hidden = 784, 256, 512

# input for encoder

x =  Input(shape=(input,))

# encoder model

hq = Dense(latent, activation='relu', kernel_initializer='glorot_normal')(x)
mu = Dense(latent, kernel_initializer='glorot_normal')(hq)
sigma = Dense(latent, kernel_initializer='glorot_normal')(hq)

# reparameterization

z = Lambda(lambda n: n[0] + exp(n[1] / 2) * random_normal(shape(n[0]), mean=0, stddev=1))([mu, sigma])

# decoder model

hz = Dense(hidden, input_dim=latent, activation='relu', kernel_initializer='glorot_normal')(z)
y = Dense(input, activation='sigmoid', kernel_initializer='glorot_normal')(hz)

# VAE model

VAE = Model(x, y)

# reconstruction loss

reconstruction_loss = -sum(x * log(y) + (1 - x) * log(1 - y), 1)

# KL divergence

KL_div = -0.5 * sum(1 + sigma - square(mu) - exp(sigma), 1)

# combined loss

loss = mean(reconstruction_loss + KL_div)

# optimizer

Adam = tf.train.AdamOptimizer
opt = Adam(0.0001, 0.99)
min = opt.minimize(loss)

# initializer

init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

# session

with tf.Session() as sess:

    sess.run(init_global)
    sess.run(init_local)

    _batch = zeros((batch_size, input), dtype=float)

    for epoch in range(epochs):

        avg = 0.0

        for batch in range(batches):

            for sample in range(batch_size):

                _batch[sample] = X[randint(0, samples)]

            _, batch_loss = sess.run([min, loss], feed_dict={x : _batch})

            avg += batch_loss

        avg /= batches

        print(avg)
























