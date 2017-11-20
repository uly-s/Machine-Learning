import numpy as np
from scipy.special import expit

# imported functions
def sigmoid(z):
    return expit(z)

def sigmoid_prime(z):
    return expit(z) * (1 - expit(z))

def tanh(z):
    return np.tanh(z)

def dot(a, b):
    return np.dot(a, b)



class RNN:
    """A recurrent Neural Network with LSTM"""

    def __init__(self, cell_count, xdim):
        """Pass the memory cell count and the input dimension,
            parameters are as follows:
            g = input nodes
            i = input gate
            f = forget gate
            o = output gate
            Wvar and Bvar are weights and biases
            dvar is delta variable, concat length
            is cell count + xdim"""

        self.cell_count = cell_count
        self.xdim = xdim
        self.concat_len = cell_count + xdim

        # weights
        self.Wg = np.random.rand(cell_count, self.concat_len) * 0.1
        self.Wi = np.random.rand(cell_count, self.concat_len) * 0.1
        self.Wf = np.random.rand(cell_count, self.concat_len) * 0.1
        self.Wo = np.random.rand(cell_count, self.concat_len) * 0.1

        # biases
        self.bg = np.random.randn(cell_count) * 0.1
        self.bi = np.random.randn(cell_count) * 0.1
        self.bf = np.random.randn(cell_count) * 0.1
        self.bo = np.random.randn(cell_count) * 0.1

        # derivative of parameters
        # weights
        self.dWg = np.zeros((cell_count, self.concat_len))
        self.dWi = np.zeros((cell_count, self.concat_len))
        self.dWf = np.zeros((cell_count, self.concat_len))
        self.dWo = np.zeros((cell_count, self.concat_len))

        # bias
        self.dbg = np.zeros(cell_count)
        self.dbi = np.zeros(cell_count)
        self.dbf = np.zeros(cell_count)
        self.dbo = np.zeros(cell_count)


    def update(self, LR):
        """Apply derivative to parameters, pass the learning rate"""

        # update weights
        self.Wg -= LR * self.dWg
        self.Wi -= LR * self.dWi
        self.Wf -= LR * self.dWf
        self.Wo -= LR * self.dWo

        # update bias
        self.bg -= LR * self.dbg
        self.bi -= LR * self.dbi
        self.bf -= LR * self.dbf
        self.bo -= LR * self.dbo

        # reset derivatives
        # weights
        self.dWg = np.zeros_like(self.Wg)
        self.dWi = np.zeros_like(self.Wi)
        self.dWf = np.zeros_like(self.Wf)
        self.dWo = np.zeros_like(self.Wo)

        # bias
        self.dbg = np.zeros_like(self.bg)
        self.dbi = np.zeros_like(self.bi)
        self.dbf = np.zeros_like(self.bf)
        self.dbo = np.zeros_like(self.bo)

class State:
    """Holds a state of the network"""

    def __init__(self, mem_cells):
        """ pass the number of memory cells"""

        self.g = np.zeros(mem_cells)
        self.i = np.zeros(mem_cells)
        self.f = np.zeros(mem_cells)
        self.o = np.zeros(mem_cells)
        self.s = np.zeros(mem_cells)
        self.h = np.zeros(mem_cells)
        self.ds = np.zeros_like(self.s)
        self.dh = np.zeros_like(self.s)

class Node:
    """ A single long short term memory node"""

    def __init__(self, net, state):
        """Pass the network parameters (net) and the network state (memory object)"""

        self.net = net
        self.state = state

        # concatenated input
        self.xc = None

    def feedforward(self, x, s0 = None, h0 = None):
        """ Feed pattern forward through network
            concatenated with previous hidden state h0 """

        if s0 is None: s0 = np.zeros_like(self.state.s)
        if h0 is None: h0 = np.zeros_like(self.state.h)

        # concatenate input
        xc = np.hstack(x, h0)

        # input state g = tanh of dot input g weights and xc plus input g bias
        # input gate i = sigmoid of dot of input gate weights times xc plus input bias bi
        # forget gate f = sigmoid of dot of forget gates times xc plus forget bias
        # output gate o = sigmoid of dot of output weights times xc plus output bias
        # new state s = g * i + s0 * f
        # new state h = new s * o

        self.state.g = tanh(dot(self.net.Wg, xc) + self.net.bg)
        self.state.i = sigmoid(dot(self.net.Wi, xc) + self.net.bi)
        self.state.f = sigmoid(dot(self.net.Wf, xc) + self.net.bf)
        self.state.o = sigmoid(dot(self.net.Wo, xc) + self.net.bo)
        self.state.s = self.state.g * self.state.i + s0 * self.state.f
        self.state.h = self.state.s * self.state.o

        # assign new xc
        self.xc = xc

    def backprop(self, ds, dh):
        """ Backpropagate through a single node
            needs the difference in the s and h states
            calculates updates to parameters """

        


class LSTM:
    """ The network with long short term memory"""

    def __init__(self, net):
        """ Pass a RNN object as seen above"""

        self.RNN = net
        self.Nodes = []
        self.inputs = []


