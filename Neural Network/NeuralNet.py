#from numba import cuda, jit
import numpy
import random

from scipy.special import expit

# simple deep neural network with parallel processing, a MSE cost function

class NeuralNet:

  # pass a tuple or list of integers
  def __init__(self, sizes):

    self.sizes = sizes

    self.layers = len(sizes)

    self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]] 

    self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


  # feed input forward, a should be vector or numpy array
  def feedforward(self, a):
    """ the output of the network is the dot product of input a and weights 
        plus the biases, because we are passing a vector or numpy array
        numpy automatically applies a vector operation elementwise """
    
    for b, w in zip(self.biases, self.weights):
      
      a = sigmoid(numpy.dot(w, a) + b)

    return a

  # backpropagate error through the network
  def backpropagate(self, x, y):
    """ Pass a pattern x and a target y and be returned a tuple
        of the cost-error gradient for the biases and weights
        in the form of layer by layer lists of numpy arrays,
        tuple is (nabla_bias, nabla_weight) """

    # nabla bias
    nabla_b = [numpy.zeros(b.shape) for b in self.biases]

    # nabla weights
    nabla_w = [numpy.zeros(w.shape) for w in self.weights]

    # activation
    activation = x

    # list of activations
    activations = [x]

    # list to store z vectors
    zs = []

    # take the dot (weighted sum) of b and w and activate it
    # add the sum to zs and the activation to activations
    for b, w in zip(self.biases, self.weights):

      # take the dot product of the activation times the weights plus biases
      z = numpy.dot(w, activation) + b

      # add to zs 
      zs.append(z)

      # set activation to sigmoid (z)
      activation = sigmoid(z)

      # append to activations
      activations.append(activation)

    # now we start the backward pass at the bottom layer

    # delta, derivative of cost for output activation times the  
    # derivative of sigmoid function for the weighted sum
    # of the output layer
    delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1])

    # gradient of bias in last layer = delta
    nabla_b[-1] = delta

    # gradient of weight in last layer = dot product of delta and the
    # transpose of the activations in the second to last layer
    nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

    # l = 1 = last layer, l = 2 = second to last
    for l in range(2, self.layers):

      # get weighted activation
      z = zs[-l]

      # get sigmoid prime of the z
      sp = sigmoid_prime(z)

      # delta = the dot product of weights and delta times sigmoid prime
      delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * sp

      # gradient of bias in layer = delta
      nabla_b[-l] = delta

      # gradient of weights in layer = dot product of activations at layer and delta
      nabla_w[-l] = numpy.dot(delta, activations[-l - 1].transpose())

    return (nabla_b, nabla_w)



  # stochastic gradient descent
  def SGD(self, training_data, epochs, LR, mini_batch_size, test_data = None):
    """ Train the neural network with tuples of patterns and target
        output, training data should be a list of tuples, epochs an int 
        specifing the number of epochs, LR = Learning rate,
        mini batch size is self explanatory, test data
        is validation data for use after each batch """

    # if test data we need the length 
    if test_data: n_test = len(test_data)

    # get length of training data
    n = len(training_data)

    # for each epoch
    # shuffle the training data
    # create a list of mini batches
    for i in range(epochs): 

      # shuffle
      random.shuffle(training_data)

      # make a list of mini batches
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

      # for each mini batch
      # call update mini batch which calls backpropagate
      for mini_batch in mini_batches:
        
        # update mini batch
        self.update_mini_batch(mini_batch, LR)

      # if we have test data process it
      if test_data:

        print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))

      else:

        print("Epoch {0} complete".format(i))

  # update mini batch
  def update_mini_batch(self, mini_batch, LR):
    """ mini batch should be a list of tuples of training examples
        , LR is learning rate. Function descends gradient via
        backpropagation and updates weights and biases accordingly """

    # create lists of numpy arrays for the error gradients of the weight and bias
    nabla_b = [numpy.zeros(b.shape) for b in self.biases]
    nabla_w = [numpy.zeros(w.shape) for w in self.weights]

    # for each tuple in mini batch, backpropagate the output to get the change
    # in the gradient delta nabla, then apply this to update nabla b, w
    for x, y in mini_batch:

      # backpropagate
      delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)

      # add delta nabla to nabla for weights and biases
      nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    # now we can update the weights and biases

    # change in weight = weight - (learning rate / mini batch size) * gradient of weight
    self.weights = [w - (LR / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

    # change in bias = bias - (learning rate / mini batch size) * gradient of bias
    self.biases = [b - (LR / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

  # validate results
  def evaluate(self, test_data):
      """ Returns the number of inputs in test data labeled correctly
          test data should be a list of tuples of patterns and targets """

      results = [(numpy.argmax(self.feedforward(x)), y) for (x, y) in test_data]

      return sum(int(x == y) for (x, y) in results)

  # derivative of mean squared / quadratic cost function
  def cost_prime(self, output, y):
    """ return a vector of partial derivatives Cx / a 
        for output activation """

    return (output - y)






# sigmoid function
def sigmoid(z):
  
  # expit is logistic function
  return expit(z)

# sigmoid prime
def sigmoid_prime(z):

  return sigmoid(z) * (1 - sigmoid(z))


# return a list of tuples of input and desired output
def loadTrainingData(entries):

        # declare an empty list of data
    
    tuples = []

    for (pattern, target, index) in zip(open("mnist_train.txt", "r"), open("mnist_train_targets.txt", "r"), range(0, entries)):

      y = numpy.reshape(numpy.array(pattern.split(), dtype = float), (784, 1))

      x = numpy.zeros((10, 1))

      z = target.split()

      for i in range(0, 10):

        x[i] = z[i]

      tuples.append((y, x))   

    return tuples

# returns tuple of patterns and labels
def loadTestData(entries):

  tuples = []

  for (pattern, label) in zip(open("mnist_test_data.txt", "r"), open("mnist_test_labels.txt", "r")):

    x = pattern.split()

    y = numpy.zeros((784, 1))

    for j in range(0, 784):

      y[j] = x[j]

    tuples.append((y, int(label)))

  
  return tuples

