#from numba import cuda, jit
import numpy

from scipy.special import expit

# simple deep neural network with parallel processing, a MSE cost function

class NeuralNet:

  # pass a tuple or list of integers
  def __init__(self, sizes):

    self.sizes = sizes

    self.layers = len(sizes)

    self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]] 

    self.weights = [numpy.random.randn(x, y) for y, x in zip(sizes[:-1], sizes[1:])]





  # feed input forward, a should be vector or numpy array
  def feedforward(self, a):

    # the output of the network is the dot product of input a and weights 
    # plus the biases, because we are passing a vector or numpy array
    # numpy automatically applies a vector operation elementwise
    
    for b, w in zip(self.biases, self.weights):
      
      a = sigmoid(numpy.dot(w, a) + b)

    return a

  # stochastic gradient descent



# sigmoid function
def sigmoid(z):
  
  return expit(z)

# load mnist training data
def loadData():

    # declare an empty list of data
    data = []
    
    for line in open("mnist_data.txt", "r"):
      
      x = line.split()

      y = [float(val) for val in x]

      data.append(y)

    return data

# load training labels
def loadLabels():

    # declare an empty list
    labels = []

    for line in open("mnist_labels.txt", "r"):

        x = line

        y = int(x)

        labels.append(y)

    return labels

# load target patterns
def loadTargets(labels):

    # declare an empty list
    targets = []

    for label in labels:

        target = []

        for index in range(0, 9):

            if label == index:
                
                target.append(1)
            
            else:

                target.append(0)

    targets.append(target)






