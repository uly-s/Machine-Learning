import numpy as np




class RNN:
  """ Character level Recurrent Neural Network trained by BPTT """

  def __init__(self, h, n):
    """ pass the size of the hidden layers and the size of vocabulary """
    self.h = h
    self.n = n

    self.Wx = np.random.randn(h, n) * 0.01 # connection from every vocab word to every hidden node
    self.Wh = np.random.randn(h, h) * 0.01 # connection from each hidden node to each hidden node
    self.Wy = np.random.randn(n, h) * 0.01 # connection from each hidden node to each vocab word

    self.bh = np.zeros((h, 1)) # biases shared between layers
    self.by = np.zeros((n, 1)) # one output bias per vocab letter

    # no replacement for real LSTM but this is a vanilla implementation
    self.mWx = np.zeros((h, n))
    self.mWh = np.zeros((h, h))
    self.mWy = np.zeros((n, h))
    self.mbh = np.zeros((h, 1)) # biases shared between layers
    self.mby = np.zeros((n, 1))


  def sample(self, hidden, seed, length):
    """ sample a sequence of integers
        from the model, hidden is the
        previous state, seed is the first
        letter, length is the number to return """

    # create a vocab size array, encode the seed index
    x = np.zeros((self.n, 1))
    x[seed] = 1

    # create a holder for sequences
    sequence = []

    # for n inputs
    for t in range(length):

      # hidden state
      hidden = np.tanh(np.dot(self.Wx, x) + np.dot(self.Wh, hidden) + self.bh)

      # output state
      output = np.dot(self.Wy, hidden) + self.by

      logit = np.exp(output)

      # probabilities, softmax
      probability = logit / np.sum(logit)

      # random index
      index = np.random.choice(range(self.n), p = probability.ravel())

      # reset x
      x = np.zeros((self.n, 1))

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
    xs, hs, ys, ps = {}, {}, {}, {}

    # loss
    loss = 0

    # set hidden state at the end to be the last hidden state
    hs[-1] = np.copy(previous)

    # FORWARD PASS
    for t in range(len(inputs)):

      # get input state
      xs[t] = np.zeros((self.n, 1))

      # set input state [t] to position t of inputs
      xs[t][inputs[t]] = 1

      # hidden state
      hs[t] = np.tanh(np.dot(self.Wx, xs[t]) + np.dot(self.Wh, hs[t-1]) + self.bh)

      # output state
      ys[t] = np.dot(self.Wy, hs[t]) + self.by

      # probabilities, softmax
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

      # cross entropy loss
      loss += -np.log(ps[t][targets[t], 0])

    # holder for the gradient
    dWx = np.zeros((self.h, self.n))
    dWh = np.zeros((self.h, self.h))
    dWy = np.zeros((self.n, self.h))
    dbh = np.zeros((self.h, 1))
    dby = np.zeros((self.n, 1))

    # next hidden state
    dn = np.zeros_like(hs[0])

    # BACKWARDS PASS
    for t in reversed(range(len(inputs))):

      # get delta output probabilities
      do = np.copy(ps[t])

      # backpropagate into the output nodes
      do[targets[t]] -= 1

      # delta output weights += dot product of delta output and the hidden state transpose
      dWy += np.dot(do[t], hs[t].T)

      # add delta output to delta bias output
      dby += do

      # backpropagate into hidden, change in 
      # hidden = dot product of the output weights transposed and delta output
      dh = np.dot(self.Wy.T, do) + dn

      # backpropagate through tnh nonlinearity
      dh_ = (1 - hs[t] * hs[t]) * dh

      # update change in bias to raw
      dbh += dh_

      # update to input weights is the dot product raw and the input state transposed
      dWx += np.dot(dh_, xs[t].T)

      # update to hidden weights is the dot product of raw of the last hidden state transposed
      dWh += np.dot(dh_, hs[t - 1].T)

      # update change in next hidden state to the dot product of the hidden weights transposed and raw
      dn = np.dot(self.Wh.T, dh_)

    # Clip parameters to mitigate exploding gradient
    for d in [dWx, dWh, dWy, dbh, dby]:
      np.clip(d, -5, 5, out = d)

    # return loss, gradient, and the last hidden state
    return loss, dWx, dWh, dWy, dbh, dby, hs[len(inputs)-1]
    

  def bptt(self, file, sequence_length, sequences):
    """ Pass a text file in read mode, the length of the 
        sequences you want to work, and the number of sequences
        to train over """

    # get a list of chars
    chars = list(set(file))
    # get size of the data
    data_size = len(file)
    # get size of vocab, should match vocab_size
    n = len(chars)

    # check for correct parameters
    if self.n != n:
      print("PARAMETER MISMATCH, INCORRECT NETWORK SIZE FOR INPUT SIZE")
    
    # encode the chars in the text file as dictionaries of integers and vice versa
    char2int = {ch:i for i,ch in enumerate(chars)}
    int2char = {i:ch for i,ch in enumerate(chars)}

    # iteration counter, data index, loss, and smoothed out loss
    iteration = 0
    index = 0
    loss = 0
    smooth_loss = -np.log(1.0/self.n) * sequence_length

    # create objects for the change in weight and bias
    dWx = np.zeros((self.h, self.n))
    dWh = np.zeros((self.h, self.h))
    dWy = np.zeros((self.n, self.h))
    dbh = np.zeros((self.h, 1))
    dby = np.zeros((self.n, 1))

    # while we still have training to do
    while iteration < sequences:

      # prepare inputs, sweeping left to right through data sequence length at a time
      # if we are at the end of the data or on the first iteration
      if index + sequence_length + 1 >= len(file) or iteration == 0:

        # previous hidden state
        previous = np.zeros((self.h, 1))

        # reset index
        index = 0

      # make integer lists out of the chars at index through seq length * 2
      inputs = [char2int[ch] for ch in file[index:index+sequence_length]]
      targets = [char2int[ch] for ch in file[index+1:index+sequence_length+1]]

      # output a sample now and then
      if iteration % 100 == 0:
        example = self.sample(previous, inputs[0], 150)
        text = ''.join(int2char[i] for i in example)
        print(text)

      # get the loss, gradient, and the next hidden state
      loss, dWx, dWh, dWy, dbh, dby, previous = self.loss(inputs, targets, previous)

      # get smooth loss
      smooth_loss = smooth_loss * 0.999 + loss * 0.001

      # print out the loss every once in a while
      if iteration % 100 == 0:
        print(smooth_loss)

      # update parameters and memory with delta 
      for par, dpar, mem in zip([self.Wx, self.Wh, self.Wy, self.bh, self.by],
                                 [dWx, dWh, dWy, dbh, dby],
                                 [self.mWx, self.mWh, self.mWy, self.mbh, self.mby]):

        # update memory to square of parameter
        mem += dpar * dpar

        # update parameter to - learning rate * par / square root of memory + 1 * 10^-8
        par += - 0.1 * dpar / np.sqrt(mem + 1e-8)





      iteration += 1
      index += sequence_length