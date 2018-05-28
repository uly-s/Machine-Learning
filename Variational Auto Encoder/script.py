#from __future__ import print_function

from keras.callbacks import LambdaCallback

import numpy
import VarAutoEncoder

data_file = open("mnist_data.txt", 'r', encoding='utf-8')

data = []

for line in data_file:
    entry = numpy.array(line.split(), dtype=float)
    data.append(entry)

x = numpy.zeros((len(data), 784))

for i, entry in enumerate(data):
    x[i] = entry

x.reshape((-1, 784))

VAE = VarAutoEncoder.VariationalAutoEncoder(784, 128, 128)

print("compiled")

VAE.fit(x,
        x,
        batch_size=64,
        epochs = 10,)

