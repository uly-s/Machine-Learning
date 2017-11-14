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


  def sample(self, hidden, seed, n):
    """ sample a sequence of integers
        from the model, hidden is the
        previous state, seed is the first
        letter, n is the number to return """

    # create a vocab size array, encode the seed index
    x = np.zeros((self.size_vocab, 1))
    x[seed] = 1

    # create a holder for sequences
    sequence = []

    # for n inputs
    for t in range(n):

      # hidden state
      hidden = np.tanh(np.dot(self.weights.input, x) + np.dot(self.weights.hidden, hidden) + self.bias.hidden)

      # output state
      output = np.dot(self.weights.output, hidden) + self.bias.output

      # probabilities
      probability = np.exp(output) / np.sum(np.exp(output))

      # random index
      index = np.random.choice(range(self.size_vocab), p = probability.ravel())

      # reset x
      x = np.zeros((self.size_vocab, 1))

      # set index
      x[index] = 1

      # append to sequence
      sequence.append(index)

    return sequence 



  def loss(self, inputs, targets, previous):
    """ Pass a list of integers as input, recieve 
        the loss, parameter updates, and last hidden
        state (loss, input, hidden, output weights, hidden and output bias, last state) """

    # state of input, hidden, output, and probabilities 
    input_state, hidden_state, output_state, probabilities = {}, {}, {}, {}

    # loss
    loss = 0

    # set hidden state at the end to be the last hidden state
    hidden_state[-1] = np.copy(previous)

    # FORWARD PASS
    for t in range(len(inputs)):

      # get input state
      input_state[t] = np.zeros((self.size_vocab, 1))

      # set input state [t] to position t of inputs
      input_state[t][inputs[t]] = 1 

      # hidden state
      hidden_state[t] = np.tanh(np.dot(self.weights.input, input_state[t]) + np.dot(self.weights.hidden, hidden_state[t-1]) + self.bias.hidden)

      # output state
      output_state[t] = np.dot(self.weights.output, hidden_state[t]) + self.bias.output

      # probabilities
      probabilities[t] = np.exp(output_state[t]) / np.sum(np.exp(output_state[t]))

      # softmax, cross entropy loss
      loss += -np.log(probabilities[t][targets[t]], 0)


    # holder for the gradient
    delta = Memory(self.size_hidden, self.size_vocab)

    # BACKWARDS PASS
    for t in reversed(range(len(inputs))):

      # get output probabilities
      deltaY = np.copy(probabilities[t])

      # backpropagate into the hidden nodes
      deltaY[targets[t]] -= -1



    # return loss, gradient, and the last hidden state
    return loss, delta, hidden_state[len(inputs)-1]
    

  def bptt(self, file, sequence_length, sequences):
    """ Pass a text file in read mode, the length of the 
        sequences you want to work, and the number of sequences
        to train over """

    # get a list of cars
    chars = list(set(file))
    # get size of the data
    data_size = len(file)
    # get size of vocab, should match vocab_size
    vocab_size = len(chars)

    # check for correct parameters
    if self.size_vocab != vocab_size:
      print("PARAMETER MISMATCH, INCORRECT NETWORK SIZE FOR INPUT SIZE")
    
    # encode the chars in the text file as dictionaries of integers and vice versa
    char2int = {ch:i for i,ch in enumerate(chars)}
    int2char = {i:ch for i,ch in enumerate(chars)}

    

    # iteration counter, data index, loss, and smoothed out loss
    iteration = 0
    index = 0
    loss = 0
    smooth_loss = -np.log(1.0/self.size_vocab) * sequence_length

    # create object for the change in weight and bias
    delta = Memory(self.size_hidden, self.size_vocab)

    # while we still have training to do
    while iteration < sequences:

      # prepare inputs, sweeping left to right through data sequence length at a time
      # if we are at the end of the data or on the first iteration
      if iteration + sequence_length + 1 >= len(file) or iteration == 0:

        # previous hidden state
        previous = np.zeros((self.size_hidden, 1))

        # reset index
        index = 0

      # make integer lists out of the chars at index through seq length * 2
      inputs = [char2int[ch] for ch in file[index:index + sequence_length]]
      targets = [char2int[ch] for ch in file[index:index + sequence_length]]

      # output a sample now and then
      if iteration % 100 == 0:
        example = self.sample(previous, inputs[0], 150)
        text = ''.join(int2char[i] for i in example)
        #print(text)

      # get the loss, gradient, and the next hidden state
      loss, delta, previous = self.loss(inputs, targets, previous)

      # get smooth loss
      smooth_loss = smooth_loss * 0.999 + loss * 0.01

      # print out the loss every once in a while
      if iteration % 100 == 0:
        print(smooth_loss)





      iteration += 1
      index += sequence_length

    




