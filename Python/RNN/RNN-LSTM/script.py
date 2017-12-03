import sys

import numpy

def random():
    return numpy.random.randint(0, 2147483647, dtype= int)



import LSTM



retData, vocab, outputs, output_size, data = LSTM.getData(open("trumptweetssample.txt", 'r', encoding='utf8').read())

RNN = LSTM.RNN(vocab, vocab, output_size, outputs, 0.01)