# Import logistic function that doesn't explode outside a 64 bit float
from scipy.special import expit as sigmoid
from numpy import zeros
from numpy import zeros_like
from numpy.random import randn
from numpy import tanh
from numpy import exp
from numpy import sum
from numpy import dot
from numpy import sqrt
from numpy import log
from numpy import atleast_2d as _2d
from numpy import column_stack as stack
from numpy import vstack
from numpy import hstack
from numpy import append
from numpy import where
from numpy import asarray
from numpy import clip
from numpy import argmax
from numpy import concatenate as concat
from numpy import copy

# derivative of sigmoid function
def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

# derivative of hyperbolic tangent
def dtanh(z):
    return 1 - tanh(z) ** 2

# probability function
def softmax(z):
    return exp(z) / sum(exp(z))

# cross entropy loss
def cross_ent(p, y):
    return -log(p[y])


# RNN class
class RNN:

    def __init__(self, n, d):
        """Pass input size (n) and number of memory cells (d)"""
        self.n = n
        self.d = d
        self.z, z = n + d, n + d

        self.x = []

        self.Cells = [Cell(n, d, self)]

        self.Wi, self.Wf, self.Wo, self.Wc, self.Wy = randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(d, n) / sqrt(d / 2)
        self.bi, self.bf, self.bo, self.bc, self.by = randn(d, 1), randn(d, 1), randn(d, 1), randn(d, 1), randn(n, 1)
        self.dWi, self.dWf, self.dWo, self.dWc, self.dWy = zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((d, n))
        self.dbi, self.dbf, self.dbo, self.dbc, self.dby = zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((n, 1))

    def FeedForward(self, inputs, ht_, ct_):

        n, d = self.n, self.d
        Cells = self.Cells

        while len(Cells) < len(inputs):
            Cells.append(Cell(n, d, self))


        for i in range(len(Cells)):
            ht_, ct_ = Cells[i].feedforward(inputs[i], ht_, ct_)

        return ht_, ct_



    def BPTT(self, outputs, ht1, ct1):

        n, d, z = self.n, self.d, self.n + self.d
        Cells = self.Cells

        loss = 0

        ht1, ct1 = zeros((d, 1)), zeros((d, 1))

        for i in reversed(range(len(outputs))):
            ht1, ct1 = Cells[i].backpropagate(outputs[i], ht1, ct1)
            loss += 0.1 * cross_ent(Cells[i].p, outputs[i])


        return loss, ht1, ct1


    def train(self, inputs, outputs, seq_length):

        n, d, z = self.n, self.d, self.n + self.d

        index = 0

        LR = 0.1

        ht_, ct_ = zeros((d, 1)), zeros((d, 1))
        ht1, ct1 = zeros((d, 1)), zeros((d, 1))



        while index < len(outputs):
            xlist = inputs[index:index + seq_length]
            ylist = outputs[index:index + seq_length]
            ht_, ct_ = self.FeedForward(xlist, ht_, ct_)
            loss, ht1, ct1 = self.BPTT(ylist, ht1, ct1)
            ht1, ct1 = zeros((d, 1)), zeros((d, 1))
            print(loss)
            self.update(LR)
            index += seq_length


    def update(self, LR):

        n, d, z = self.n, self.d, self.n + self.d

        self.Wi -= LR * self.dWi
        self.Wf -= LR * self.dWf
        self.Wo -= LR * self.dWo
        self.Wc -= LR * self.dWc

        #print(self.Wy.shape)
        #print(self.dWy.shape)

        self.Wy -= LR * self.dWy

        self.bi -= LR * self.dbi
        self.bf -= LR * self.dbf
        self.bo -= LR * self.dbo
        self.bc -= LR * self.dbc
        self.by -= LR * self.dby

        self.dWi, self.dWf, self.dWo, self.dWc, self.dWy = zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((d, n))
        self.dbi, self.dbf, self.dbo, self.dbc, self.dby = zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((n, 1))




class Cell:

    def __init__(self, n, d, rnn):
        """Pass the input size (n) and memory cell size (d), create hidden state of size d"""
        self.n, self.d, self.h, self.z, z = n, d, zeros((d, 1)), n + d, n + d
        self.rnn = rnn


    def feedforward(self, x, c_, h_):
        """Pass an input of size n, the previous hidden state, and the previous cell state"""
        n, d = self.n, self.d
        Wi, Wf, Wo, Wc, Wy = self.rnn.Wi, self.rnn.Wf, self.rnn.Wo, self.rnn.Wc, self.rnn.Wy
        bi, bf, bo, bc, by = self.rnn.bi, self.rnn.bf, self.rnn.bo, self.rnn.bc, self.rnn.by

        # one hot encoding
        index = x
        x = zeros((n, 1))
        x[index] = 1

        # input g is input x + previous hidden state
        g = concat((x, h_))

        # gate activations
        it = sigmoid(dot(Wi.T, g) + bi)
        ft = sigmoid(dot(Wf.T, g) + bf)
        ot = sigmoid(dot(Wo.T, g) + bo)

        # non linearity activation
        ct = tanh(dot(Wc.T, g) + bc)

        # cell state
        c = ft * c_ + it * ct

        # squashed hidden state
        ht = ot * tanh(c)

        # get output state
        yt = dot(Wy.T, ht) + by

        # call softmax, get probability
        p = softmax(yt)

        self.c_, self.h_ = c_, h_
        self.it, self.ft, self.ot, self.ct = it, ft, ot, ct
        self.c, self.ht, self.yt, self.p, self.g = c, ht, yt, p, g

        return ht, c


    def backpropagate(self, y, ht1, ct1):

        n, d = self.n, self.d
        Wi, Wf, Wo, Wc, Wy = self.rnn.Wi, self.rnn.Wf, self.rnn.Wo, self.rnn.Wc, self.rnn.Wy
        dWi, dWf, dWo, dWc, dWy = self.rnn.dWi, self.rnn.dWf, self.rnn.dWo, self.rnn.dWc, self.rnn.dWy
        dbi, dbf, dbo, dbc, dby = self.rnn.dbi, self.rnn.dbf, self.rnn.dbo, self.rnn.dbc, self.rnn.dby
        c_, h_ = self.c_, self.h_
        it, ft, ot, ct = self.it, self.ft, self.ot, self.ct
        c, ht, yt, p = self.c, self.ht, self.yt, self.p
        g = self.g

        dy = p.copy()
        dy[y] -= 1

        dh = dot(Wy, dy) + ht1

        do = tanh(c) * dh
        do = dsigmoid(ot) * do

        dc = ot * dh * dtanh(c)
        dc = dc + ct1

        df = c_ * dc
        df = dsigmoid(ft) * df

        di = ct * dc
        di = dsigmoid(it) * di

        dct = it * dc
        dct = dtanh(ct) * dct

        dWf += dot(g, df.T)
        dWi += dot(g, di.T)
        dWo += dot(g, do.T)
        dWc += dot(g, dc.T)
        dWy += dot(ht, dy.T)

        dbf += df
        dbi += di
        dbo += do
        dbc += dc
        dby += dy

        dxi = dot(Wi, df)
        dxf = dot(Wf, di)
        dxo = dot(Wo, do)
        dxc = dot(Wc, dct)

        dx = dxf + dxi + dxo + dxc

        dht1 = dx[n:]
        dct1 = ft * dc

        return dht1, dct1








































