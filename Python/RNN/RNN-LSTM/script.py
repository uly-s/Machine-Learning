import sys

import RNN

import numpy

def random():
    return numpy.random.randint(0, 2147483647, dtype= int)

file = open('binaryints.txt', 'w')


for i in range(0, 10):
    x = random()
    file.write('{0:010} {1}\n'.format(x, bin(x)[2:].zfill(32)))


file.close()

