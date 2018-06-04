from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Flatten, Dense, Lambda, LeakyReLU, Dropout, BatchNormalization, GaussianNoise
from keras.optimizers import Adam
from numpy import prod, add, array, zeros, ones
from numpy.random import normal, randint


def Generator(latent, img_shape, hidden_sizes=[256, 512, 1024], dropout = 0.4, momentum=0.8, alpha=0.2):
    """Returns sequential generator model, takes in latent dim, image shape = (int, int), hidden sizes = [int, int, int]
       dropout, momentum for batch normalization, and alpha for leaky relu"""

    G = Sequential()

    # first hidden layer

    G.add(Dense(hidden_sizes[0], input_dim=latent))
    G.add(LeakyReLU(alpha=alpha))
    G.add(BatchNormalization(momentum=momentum))
    G.add(Dropout(dropout))

    # second

    G.add(Dense(hidden_sizes[1]))
    G.add(LeakyReLU(alpha=alpha))
    G.add(BatchNormalization(momentum=momentum))
    G.add(Dropout(dropout))

    # third

    G.add(Dense(hidden_sizes[2]))
    G.add(LeakyReLU(alpha=alpha))
    G.add(BatchNormalization(momentum=momentum))
    G.add(Dropout(dropout))

    # final hidden layer, dependent on image shape

    G.add(Dense(prod(img_shape), activation='tanh'))

    # get testing layer
    y = Dense(1, activation='sigmoid')(G)

    # reshape

    G.add(Reshape(img_shape))

    # summary

    G.summary()

    return G, y

def Discriminator(input_size, img_shape, hidden_sizes=[512, 256], dropout=0.4, momentum=0.8, alpha=0.2):
    """ Feed image shape = (int, int), hidden sizes = [int, int], dropout, momentum for batch normal, and alpha for
        leaky relu """

    D = Sequential()

    # flatten input

    D.add(Flatten(input_shape=img_shape))

    # first hidden layer

    D.add(Dense(hidden_sizes[0]))
    D.add(LeakyReLU(alpha=alpha))
    D.add(Dropout(dropout))

    # second

    D.add(Dense(hidden_sizes[1]))
    D.add(LeakyReLU(alpha=alpha))
    D.add(Dropout(dropout))

    # output layer

    D.add(Dense(1, activation='sigmoid'))

    # set trainable to false

    D.trainable = False

    # summary

    D.summary()

    # compile

    return D

data_file = open("mnist_data.txt", 'r', encoding='utf-8')

data = []

for line in data_file:
    entry = array(line.split(), dtype=float)
    data.append(entry)

x, samples = zeros((len(data), 784)), len(data)

for i, entry in enumerate(data):
    x[i] = entry



x = x / 127.5 - 1


latent, img_shape, input_size, k, k_init = 256, (28, 28), 784, 10, 100


G, y = Generator(latent, img_shape)
D = Discriminator(input_size, img_shape)

D.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

z = Input(shape=(latent,))

Gz = G(z)
GD = D(Gz)




GAN = Model(z, GD)
GEN = Model(z, y)

GAN.summary()
GEN.summary()


GAN.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
GEN.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')


epochs = 100
batch_size = 128
batches = int(samples / batch_size)

Gx0 = zeros((batch_size, latent))
Gx1 = zeros((batch_size, latent))
Gy = zeros((batch_size, 1))

Dx0 = zeros((batch_size, img_shape[0], img_shape[1]))
Dx1 = zeros((batch_size, img_shape[0], img_shape[1]))

Dy0 = zeros((batch_size, 1))
Dy1 = ones((batch_size, 1))

noise = normal(0, 1, (samples, latent))

# for each epoch
for epoch in range(epochs):

    print("Epoch: " + str(epoch) + '\n')

    real_loss, fake_loss, real_acc, fake_acc, D_loss, G_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    #for batch in batches
    for batch in range(batches):

        # train discriminator for k batches
        for step in range(k):

            # for each sample in the batch
            for sample in range(batch_size):

                i = randint(0, samples)

                Gx0[sample] = noise[i]

                i = randint(0, samples)

                Dx1[sample] = x[i].reshape((28, 28))

            Dx0 = G.predict(Gx0)

            Dx0_loss = D.train_on_batch(Dx0, Dy0)
            Dx1_loss = D.train_on_batch(Dx1, Dy1)

            D_loss = 0.5 * add(Dx0_loss[0], Dx1_loss[0])

            print("D loss: " + str(D_loss) + ", fake loss: " + str(Dx0_loss[0]) + ", fake acc: " + str(Dx0_loss[1]) + ", real loss: " + str(Dx1_loss[0]) + ", real acc: " + str(Dx1_loss[1]))

        # for each sample in the batch
        for sample in range(batch_size):

            i = randint(0, samples)

            Gx1[sample] = noise[i]


        pred = G.predict(Gx1)
        Gy = D.predict(pred)

        G_loss = GEN.train_on_batch(Gx1, Gy)

        #print("batch: " + str(batch) + ", gen loss: " + str(G_loss))




