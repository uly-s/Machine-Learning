# Import logistic function that doesn't explode outside a 64 bit float
from scipy.special import expit as sigmoid
from numpy import zeros, zeros_like, tanh, exp, sum, dot, sqrt, log, argmax, concatenate as concat, copy, clip
from numpy.random import randn


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

    def __init__(self, n, d, RL, LR):
        """Pass input size (n), number of memory cells (d), recurrence length (RL), and learning rate (LR)"""
        self.n, self.d, self.z, z = n, d, n + d, n + d
        self.d = d
        self.z, z = n + d, n + d
        self.RL = RL
        self.LR = LR

        self.count = 0

        self.x = []

        self.Cells = [Cell(n, d, self)]

        self.Wi, self.Wf, self.Wo, self.Wc, self.Wy = randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(d, n) / sqrt(d / 2)
        self.bi, self.bf, self.bo, self.bc, self.by = randn(d, 1), randn(d, 1), randn(d, 1), randn(d, 1), randn(n, 1)
        self.dWi, self.dWf, self.dWo, self.dWc, self.dWy = zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((d, n))
        self.dbi, self.dbf, self.dbo, self.dbc, self.dby = zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((n, 1))

    def FeedForward(self, inputs, ht_, ct_):

        n, d, rl, Cells = self.n, self.d, self.RL, self.Cells

        while len(Cells) < rl:
            Cells.append(Cell(n, d, self))

        for cell, x in zip(Cells, range(len(inputs))):
            ht_, ct_ = cell.feedforward(x, ht_, ct_)

        return ht_, ct_



    def BPTT(self, outputs):

        n, d, z, rl = self.n, self.d, self.n + self.d, self.RL
        Cells = self.Cells

        ht1, ct1 = zeros((d, 1)), zeros((d, 1))

        avg_loss = 0

        for i in reversed(range(rl)):
            ht1 = Cells[i-1].ht
            ct1 = Cells[i-1].c
            loss, ht1, ct1 = Cells[i].backpropagate(outputs[i], ht1, ct1)
            avg_loss += loss

        avg_loss /= rl

        return avg_loss


    def train(self, inputs, outputs):

        n, d, z, rl = self.n, self.d, self.n + self.d, self.RL
        index = 0
        LR = 0.1
        loss = 0

        ht_, ct_ = zeros((d, 1)), zeros((d, 1))

        while index < len(outputs):
            xlist = inputs[index:index + rl]
            ylist = outputs[index:index + rl]
            ht_, ct_ = self.FeedForward(xlist, ht_, ct_)
            loss = self.BPTT(ylist)
            print(loss)
            self.update(LR)
            index += 1

    def update(self, LR):

        n, d, z = self.n, self.d, self.n + self.d

        self.Wi -= LR * self.dWi
        self.Wf -= LR * self.dWf
        self.Wo -= LR * self.dWo
        self.Wc -= LR * self.dWc
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
        """Pass the input size (n) and memory cell size (d), create hidden state of size d, pass rnn (self)"""
        self.n, self.d, self.h, self.z, z = n, d, zeros((d, 1)), n + d, n + d
        self.rnn = rnn


    def feedforward(self, x, c_, h_):
        """Pass an input of size n, the previous hidden state(ht), and the previous cell state(c)"""
        n, d = self.n, self.d
        Wi, Wf, Wo, Wc, Wy = self.rnn.Wi, self.rnn.Wf, self.rnn.Wo, self.rnn.Wc, self.rnn.Wy
        bi, bf, bo, bc, by = self.rnn.bi, self.rnn.bf, self.rnn.bo, self.rnn.bc, self.rnn.by

        index = x       # one hot encoding
        x = zeros((n, 1))
        x[index] = 1
        g = concat((x, h_))         # input g is input x + previous hidden state

        it = sigmoid(dot(Wi.T, g) + bi)     # gate activations
        ft = sigmoid(dot(Wf.T, g) + bf)
        ot = sigmoid(dot(Wo.T, g) + bo)
        ct = tanh(dot(Wc.T, g) + bc)        # non linearity activation
        c = ft * c_ + it * ct       # cell state

        ht = ot * tanh(c)       # squashed hidden state
        yt = dot(Wy.T, ht) + by     # output state
        p = softmax(yt)     # call softmax, get probability

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

        dy = copy(p)
        dy[y] -= 1

        loss = cross_ent(p, y)

        dh = dot(Wy, dy) + ht1
        dh = clip(dh, -6, 6)

        do = tanh(c) * dh
        do = dsigmoid(ot) * do

        dc = ot * dh * dtanh(c)
        dc = dc + ct1
        dc = clip(dc, -6, 6)

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

        dxi = dot(Wi, di)
        dxf = dot(Wf, df)
        dxo = dot(Wo, do)
        dxc = dot(Wc, dct)

        dx = dxf + dxi + dxo + dxc

        dht1 = dx[n:]
        dct1 = ft * dc

        return loss, dht1, dct1








































