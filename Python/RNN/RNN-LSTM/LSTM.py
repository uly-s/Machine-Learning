# Import logistic function that doesn't explode outside a 64 bit float
from scipy.special import expit as sigmoid

from numpy import zeros
from numpy import zeros_like
from numpy.random import random
from numpy import tanh
from numpy import exp
from numpy import dot
from numpy import sqrt
from numpy import clip
from numpy import atleast_2d
from numpy import vstack
from numpy import append
from numpy import where
from numpy import asarray

# derivative of sigmoid function
def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

# derivative of hyperbolic tangent
def dtanh(z):
    return 1 - tanh(z) ** 2



# RNN class
class RNN:

    def __init__(self, input_size, output_size, vocab_size, expected_output, learning_rate):

        # input array
        self.x = zeros(input_size)

        # input length
        self.xdim = input_size

        # output array, expected next word
        self.y = zeros(output_size)

        # output size
        self.ydim = output_size

        # vocab size / recurrence length
        self.vocab = vocab_size

        # learning rate
        self.LR = learning_rate

        # weight matrix, vocab size times vocab size
        self.W = random((vocab_size, vocab_size))

        # array of inputs, vocab size times input length
        self.inputs = zeros((vocab_size + 1, input_size))

        # array of cell states, vocab size times output length
        self.cells = zeros((vocab_size + 1, output_size))

        # array of outputs, vocab size times output size
        self.outputs = zeros((vocab_size + 1, output_size))

        # array of hidden states, vocab size times output size
        self.states = zeros((vocab_size + 1, output_size))

        # input gate
        self.i = zeros((vocab_size + 1, output_size))

        # forget gate
        self.f = zeros((vocab_size + 1, output_size))

        # output gate
        self.o = zeros((vocab_size + 1, output_size))

        # cell state
        self.s = zeros((vocab_size + 1, output_size))

        # expected output values
        #self.expected = vstack((zeros(expected_output.shape[0]), expected_output.T))

        self.lstm = LSTM(input_size, output_size, vocab_size, learning_rate)


class LSTM:

    def __init__(self, input_size, output_size, vocab_size, learning_rate):

        # input
        self.x = zeros(input_size + output_size)

        # input size
        self.xdim = input_size + output_size

        # output
        self.y = zeros(output_size)

        # output size
        self.ydim = output_size

        # recurrence rate / size of vocab
        self.vocab = vocab_size

        # learning rate
        self.LR = learning_rate

        # gate weight matrices
        # input
        self.i = random((output_size, input_size + output_size))

        # forget
        self.f = random((output_size, input_size + output_size))

        # output
        self.o = random((output_size, input_size + output_size))

        # cell state
        self.s = random((output_size, input_size + output_size))

        # gradients
        # nabla input
        self.Ni = zeros_like(self.i)

        # forget
        self.Nf = zeros_like(self.f)

        # output
        self.No = zeros_like(self.o)

        # cell state
        self.Ns = zeros_like(self.s)




def getData(stream):

    data = stream

    text = list(data)

    outputSize = len(text)

    data = list(set(text))

    uniqueWords, dataSize = len(data), len(data)

    returnData = zeros((uniqueWords, dataSize))

    for i in range(0, dataSize):
        returnData[i][i] = 1

    returnData = append(returnData, atleast_2d(data), axis=0)

    output = zeros((uniqueWords, outputSize))

    for i in range(0, outputSize):
        index = where(asarray(data) == text[i])

        output[:, i] = returnData[0:-1, index[0]].astype(float).ravel()

    return returnData, uniqueWords, output, outputSize, data





