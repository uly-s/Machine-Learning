import numpy as np

class RNN:
  """ Character level Recurrent Neural Network trained by BPTT """

  def __init__(self, size_hidden, size_vocab):
    """ pass the size of the hidden layers and the size of vocabulary """
    self.size_hidden = size_hidden
    self.size_vocab = size_vocab

    self.Wxh = np.random.randn(size_hidden, size_vocab) * 0.01 # connection from every vocab word to every hidden node
    self.Whh = np.random.randn(size_hidden, size_hidden) * 0.01 # connection from each hidden node to each hidden node
    self.Why = np.random.randn(size_vocab, size_hidden) * 0.01 # connection from each hidden node to each vocab word

    self.bh = np.zeros((size_hidden, 1)) # biases shared between layers
    self.by = np.zeros((size_vocab, 1)) # one output bias per vocab letter

    # no replacement for real LSTM but this is a vanilla implementation
    self.mWxh = np.zeros((size_hidden, size_vocab)) 
    self.mWhh = np.zeros((size_hidden, size_hidden)) 
    self.mWhy = np.zeros((size_vocab, size_hidden))
    self.mbh = np.zeros((size_hidden, 1)) # biases shared between layers
    self.mby = np.zeros((size_vocab, 1))

    

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
      hidden = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hidden) + self.bh)

      # output state
      output = np.dot(self.Why, hidden) + self.by

      logit = np.exp(output)

      # probabilities, softmax
      probability = logit / np.sum(logit)

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
      hidden_state[t] = np.tanh(np.dot(self.Wxh, input_state[t]) + np.dot(self.Whh, hidden_state[t-1]) + self.bh)

      # output state
      output_state[t] = np.dot(self.Why, hidden_state[t]) + self.by

      # probabilities, softmax
      probabilities[t] = np.exp(output_state[t]) / np.sum(np.exp(output_state[t]))

      # cross entropy loss
      loss += -np.log(probabilities[t][targets[t], 0])

    # holder for the gradient
    dWxh = np.zeros((self.size_hidden, self.size_vocab)) 
    dWhh = np.zeros((self.size_hidden, self.size_hidden)) 
    dWhy = np.zeros((self.size_vocab, self.size_hidden))
    dbh = np.zeros((self.size_hidden, 1)) 
    dby = np.zeros((self.size_vocab, 1))

    # next hidden state
    delta_next = np.zeros_like(hidden_state[0])

    # BACKWARDS PASS
    for t in reversed(range(len(inputs))):

      # get output probabilities
      delta_output = np.copy(probabilities[t])

      # backpropagate into the output nodes
      delta_output[targets[t]] -= 1

      # delta output weights += dot product of delta output and the hidden state transpose
      dWhy += np.dot(delta_output[t], hidden_state[t].T)

      # add delta output to delta bias output
      dby += delta_output

      # backpropagate into hidden, change in 
      # hidden = dot product of the output weights transposed and delta output
      delta_hidden = np.dot(self.Why.T, delta_output) + delta_next

      # backpropagate through tnh nonlinearity
      delta_hidden_raw = (1 - hidden_state[t] * hidden_state[t]) * delta_hidden

      # update change in bias to raw
      dbh += delta_hidden_raw

      # update to input weights is the dot product raw and the input state transposed
      dWxh += np.dot(delta_hidden_raw, input_state[t].T)

      # update to hidden weights is the dot product of raw of the last hidden state transposed
      dWhh += np.dot(delta_hidden_raw, hidden_state[t - 1].T)

      # update change in next hidden state to the dot product of the hidden weights transposed and raw
      delta_next = np.dot(self.Whh.T, delta_hidden_raw)

    # Clip parameters to mitigate exploding gradient
    for d in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(d, -5, 5, out = d)

    # return loss, gradient, and the last hidden state
    return loss, dWxh, dWhh, dWhy, dbh, dby, hidden_state[len(inputs)-1]
    

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

    # create objects for the change in weight and bias
    dWxh = np.zeros((self.size_hidden, self.size_vocab)) 
    dWhh = np.zeros((self.size_hidden, self.size_hidden)) 
    dWhy = np.zeros((self.size_vocab, self.size_hidden))
    dbh = np.zeros((self.size_hidden, 1)) 
    dby = np.zeros((self.size_vocab, 1))

    # while we still have training to do
    while iteration < sequences:

      # prepare inputs, sweeping left to right through data sequence length at a time
      # if we are at the end of the data or on the first iteration
      if index + sequence_length + 1 >= len(file) or iteration == 0:

        # previous hidden state
        previous = np.zeros((self.size_hidden, 1))

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
      loss, dWxh, dWhh, dWhy, dbh, dby, previous = self.loss(inputs, targets, previous)

      # get smooth loss
      smooth_loss = smooth_loss * 0.999 + loss * 0.001

      # print out the loss every once in a while
      if iteration % 100 == 0:
        print(smooth_loss)

      # update parameters and memory with delta 
      for par, dpar, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                 [dWxh, dWhh, dWhy, dbh, dby],
                                 [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):

        # update memory to square of parameter
        mem += dpar * dpar

        # update parameter to - learning rate * par / square root of memory + 1 * 10^-8
        par += - 0.1 * dpar / np.sqrt(mem + 1e-8)





      iteration += 1
      index += sequence_length