import numpy as np
from scipy.special import expit

# imported functions
def sigmoid(z):
    return expit(z)

def sigmoid_prime(z):
    return expit(z) * (1 - expit(z))

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - z ** 2

def dot(a, b):
    return np.dot(a, b)

def outer(a, b):
    return np.outer(a, b)



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
        self.dh = np.zeros_like(self.h)

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

        # save state
        self.s0 = s0
        self.h0 = h0

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

    def backprop(self, xs, xh):
        """ Backpropagate through a single node
            needs the difference in the s and h states
            calculates updates to parameters """

        ds = self.state.o * xh + xs
        do = self.state.s * xh
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s0 * ds

        dix = sigmoid_prime(self.state.i) * di
        dfx = sigmoid_prime(self.state.f) * df
        dox = sigmoid_prime(self.state.o) * do
        dgx = tanh_prime(self.state.g) * dg

        self.net.dWi += outer(dix, self.xc)
        self.net.dWf += outer(dfx, self.xc)
        self.net.dWo += outer(dox, self.xc)
        self.net.dWg += outer(dgx, self.xc)

        self.net.dbi += dix
        self.net.dbf += dfx
        self.net.dbo += dox
        self.net.dbg += dgx

        dxc = np.zeros_like(self.xc)
        dxc += dot(self.net.Wi.T, dix)
        dxc += dot(self.net.Wf.T, dfx)
        dxc += dot(self.net.Wo.T, dox)
        dxc += dot(self.net.Wg.T, dgx)

        self.state.ds = ds * self.state.f
        self.state.dh = dxc[self.net.xdim]




        


class LSTM:
    """ The network with long short term memory"""

    def __init__(self, cell_count, xdim):
        """ Pass a RNN object as seen above"""

        self.rnn = RNN(cell_count, xdim)
        self.Nodes = []
        self.inputs = []

    def FeedForward(self, x):
        """Feed pattern forward, adds to input list, makes a new node
            x should be a last of numbers"""

        # add pattern to inputs
        self.inputs.append(x)

        # add a new node for the input
        self.Nodes.append(Node(self.rnn, State(self.rnn.cell_count)))

        # base case, first input, no recurrence
        if len(self.inputs) == 1:
            self.Nodes[0].feedforward(x)
        else:
            index = len(self.inputs) - 1 # Recurrent case, the previous state is composed of all previous states
            self.Nodes[index].feedforward(x, self.Nodes[index-1].state.s, self.Nodes[index-1].state.h)


    #def BPTT(self, labels):




