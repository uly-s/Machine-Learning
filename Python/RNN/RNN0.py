import numpy as np

class Weights:
  """ Object to hold weights and simplify syntax to look less mathematical """

  def __init__(self, size_hidden, size_vocab, random):
    """ Pass the number of hidden nodes per layer and the size of the vocabulary """
    if random:
      self.input = np.random.randn(size_hidden, size_vocab) * 0.01 # connection from every vocab word to every hidden node
      self.hidden = np.random.randn(size_hidden, size_hidden) * 0.01 # connection from each hidden node to each hidden node
      self.output = np.random.randn(size_vocab, size_hidden) * 0.01 # connection from each hidden node to each vocab word
    
    else:
      self.input = np.zeros((size_hidden, size_vocab)) # connection from every vocab word to every hidden node
      self.hidden = np.zeros((size_hidden, size_hidden)) # connection from each hidden node to each hidden node
      self.output = np.zeros((size_vocab, size_hidden)) # connection from each hidden node to each vocab word


class Bias:
  """ Object to hold biases and make the syntax look less mathematical """

  def __init__(self, size_hidden, size_vocab):
    """ Pass size of hidden layer and size of vocabulary """
    self.hidden = np.zeros((size_hidden, 1)) # biases shared between layers
    self.output = np.zeros((size_vocab, 1)) # one output bias per vocab letter

class Memory:
  """ Object to store the networks memory of past states """

  def __init__(self, size_hidden, size_vocab):
    """ pass the size of the hidden layers and the size of vocabulary """
    self.weights = Weights(size_hidden, size_vocab, False)
    self.bias = Bias(size_hidden, size_vocab)

class RNN:
  """ Recurrent Neural Network with LSTM units trained by BPTT """

  def __init__(self, size_hidden, size_vocab):
    """ pass the size of the hidden layers and the size of vocabulary """
    self.size_hidden = size_hidden
    self.size_vocab = size_vocab
    self.weights = Weights(size_hidden, size_vocab, True)
    self.bias = Bias(size_hidden, size_vocab)
    self.mem = Memory(size_hidden, size_vocab)

  def feedforward(self, input):


  def loss(self, inputs, targets, previous):
    """ Pass a list of integers as input, recieve 
        the loss, parameter updates, and last hidden
        state (loss, input, hidden, output weights, hidden and output bias, last state) """



  def bptt(self, file, sequence_length, sequences):
    """ Pass a text file in read mode, the length of the 
        sequences you want to work, and the number of sequences
        to train over """



