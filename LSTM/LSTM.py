# Import logistic function that doesn't explode outside a 64 bit float
from scipy.special import expit as sigmoid
from numpy import zeros, zeros_like, tanh, exp, sum, dot, sqrt, log, argmax, concatenate as concat, copy, clip, ravel
from numpy.random import randn, choice

def dsigmoid(z): return sigmoid(z) * (1 - sigmoid(z)) # derivative of sigmoid function
def dtanh(z): return 1 - tanh(z) ** 2 # derivative of hyperbolic tangent
def softmax(z): return exp(z) / sum(exp(z)) # probability function
def cross_ent(p, y): return -log(p[y]) # cross entropy loss

def preprocess(file_name):
    """pass the name of the text file, returns size of the alphabet, a list of integer inputs, and one of outputs.
       Also returns dictionaries for encoding and decoding chars"""

    file = open(''.join(file_name), 'r', encoding='utf-8').read()
    text = list(file)
    alphabet = list(set(text))
    n = (len(alphabet))

    encode = {ch: i for i, ch in enumerate(alphabet)}
    decode = {i: ch for i, ch in enumerate(alphabet)}

    inputs = [encode[ch] for ch in text]
    outputs = [inputs[i + 1] for i in range(len(inputs) - 1)]

    return n, inputs, outputs, encode, decode

# RNN class
class RNN:

    def __init__(self, n, d, RL, encode, decode):
        """Pass input size (n), number of memory cells (d), recurrence length (RL), and ecode/decode from preprocess"""
        self.n, self.d, self.z, z, self.RL = n, d, n + d, n + d, RL
        self.encode, self.decode = encode, decode
        self.Cells = [Cell(n, d, self) for cell in range(RL)]
        self.Wi, self.Wf, self.Wo, self.Wc, self.Wy = randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(z, d) / sqrt(z / 2), randn(d, n) / sqrt(d / 2)
        self.bi, self.bf, self.bo, self.bc, self.by = zeros((d, 1)), zeros((d, 1)) + 0.1, zeros((d, 1)), zeros((d, 1)), zeros((n, 1))
        self.dWi, self.dWf, self.dWo, self.dWc, self.dWy = zeros((z, d)) + 0.1, zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((d, n))
        self.dbi, self.dbf, self.dbo, self.dbc, self.dby = zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((n, 1))

    def FeedForward(self, inputs, ht_, ct_):

        n, d, rl, Cells = self.n, self.d, self.RL, self.Cells

        for cell, x in zip(Cells, range(len(inputs))):
            ht_, ct_ = cell.feedforward(x, ht_, ct_)

        return ht_, ct_

    def BPTT(self, outputs, ht1, ct1):

        n, d, z, rl = self.n, self.d, self.n + self.d, self.RL
        Cells = self.Cells

        avg_loss = 0

        for i in reversed(range(rl)):
            ht1 = Cells[i-1].ht
            ct1 = Cells[i-1].c
            loss, ht1, ct1 = Cells[i].backpropagate(outputs[i], ht1, ct1)
            avg_loss += loss

        avg_loss /= rl

        return avg_loss, ht1, ct1

    def train(self, inputs, outputs, batches):

        n, d, z, rl = self.n, self.d, self.n + self.d, self.RL
        L, index, t, sum, batch, best, ys = 0, 0, 0, 0, 0, 100000, len(outputs)
        a, b1, b2, e = 0.001, 0.9, 0.999, 1e-8

        mWi, mWf, mWo, mWc, mWy = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        mbi, mbf, mbo, mbc, mby = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))

        vWi, vWf, vWo, vWc, vWy = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        vbi, vbf, vbo, vbc, vby = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))

        ht_, ct_ = zeros((d, 1)), zeros((d, 1))
        ht1, ct1 = zeros((d, 1)), zeros((d, 1))

        while t < batches:

            t += 1

            xlist, ylist = inputs[index:index + rl], outputs[index:index + rl]
            ht_, ct_ = self.FeedForward(xlist, ht_, ct_)
            loss, ht1, ct1 = self.BPTT(ylist, ht1, ct1)

            sum += loss
            batch+=1
            L = sum / batch

            print('t: ' + str(t) + ', L: ' + str(L) + ', l: ' + str(loss))

            dWi, dWf, dWo, dWc, dWy = self.dWi, self.dWf, self.dWo, self.dWc, self.dWy
            dbi, dbf, dbo, dbc, dby = self.dbi, self.dbf, self.dbo, self.dbc, self.dby

            mWi = b1 * mWi + (1 - b1) * dWi
            mWf = b1 * mWf + (1 - b1) * dWf
            mWo = b1 * mWo + (1 - b1) * dWo
            mWc = b1 * mWc + (1 - b1) * dWc
            mWy = b1 * mWy + (1 - b1) * dWy
            mbi = b1 * mbi + (1 - b1) * dbi
            mbf = b1 * mbf + (1 - b1) * dbf
            mbo = b1 * mbo + (1 - b1) * dbo
            mbc = b1 * mbc + (1 - b1) * dbc
            mby = b1 * mby + (1 - b1) * dby

            vWi = b2 * vWi + (1 - b2) * dWi**2
            vWf = b2 * vWf + (1 - b2) * dWf**2
            vWo = b2 * vWo + (1 - b2) * dWo**2
            vWc = b2 * vWc + (1 - b2) * dWc**2
            vWy = b2 * vWy + (1 - b2) * dWy**2
            vbi = b2 * vbi + (1 - b2) * dbi**2
            vbf = b2 * vbf + (1 - b2) * dbf**2
            vbo = b2 * vbo + (1 - b2) * dbo**2
            vbc = b2 * vbc + (1 - b2) * dbc**2
            vby = b2 * vby + (1 - b2) * dby**2

            mWi_ = mWi / (1 - b1**t)
            mWf_ = mWf / (1 - b1**t)
            mWo_ = mWo / (1 - b1**t)
            mWc_ = mWc / (1 - b1**t)
            mWy_ = mWy / (1 - b1**t)
            mbi_ = mbi / (1 - b1**t)
            mbf_ = mbf / (1 - b1**t)
            mbo_ = mbo / (1 - b1**t)
            mbc_ = mbc / (1 - b1**t)
            mby_ = mby / (1 - b1**t)

            vWi_ = vWi / (1 - b2**t)
            vWf_ = vWf / (1 - b2**t)
            vWo_ = vWo / (1 - b2**t)
            vWc_ = vWc / (1 - b2**t)
            vWy_ = vWy / (1 - b2**t)
            vbi_ = vbi / (1 - b2**t)
            vbf_ = vbf / (1 - b2**t)
            vbo_ = vbo / (1 - b2**t)
            vbc_ = vbc / (1 - b2**t)
            vby_ = vby / (1 - b2**t)

            self.Wi = self.Wi - a * mWi_ / (sqrt(vWi_) + e)
            self.Wf = self.Wf - a * mWf_ / (sqrt(vWf_) + e)
            self.Wo = self.Wo - a * mWo_ / (sqrt(vWo_) + e)
            self.Wc = self.Wc - a * mWc_ / (sqrt(vWc_) + e)
            self.Wy = self.Wy - a * mWy_ / (sqrt(vWy_) + e)
            self.bi = self.bi - a * mbi_ / (sqrt(vbi_) + e)
            self.bf = self.bf - a * mbf_ / (sqrt(vbf_) + e)
            self.bo = self.bo - a * mbo_ / (sqrt(vbo_) + e)
            self.bc = self.bc - a * mbc_ / (sqrt(vbc_) + e)
            self.by = self.by - a * mby_ / (sqrt(vby_) + e)

            self.dWi, self.dWf, self.dWo, self.dWc, self.dWy = zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((z, d)), zeros((d, n))
            self.dbi, self.dbf, self.dbo, self.dbc, self.dby = zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((d, 1)), zeros((n, 1))

            if t % 100 == 0: sum = 0; batch = 0;

            index += rl

            if index + rl >= ys: index = 0

        print(best)

class Cell:

    def __init__(self, n, d, rnn):
        """Pass the input size (n) and memory cell size (d), create hidden state of size d, pass rnn (self)"""
        self.n, self.d, self.z = n, d, n + d
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
        c, ht, yt, p, g = self.c, self.ht, self.yt, self.p, self.g

        loss = cross_ent(p, y)

        dy = copy(p)
        dy[y] -= 1
        dh = clip(dot(Wy, dy) + ht1, -6, 6)
        do = tanh(c) * dh * dsigmoid(ot)
        dc = clip((ot * dh * dtanh(c)) + ct1, -6,  6)
        df = c_ * dc * dsigmoid(ft)
        di = ct * dc * dsigmoid(it)
        dct = it * dc * dtanh(ct)

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

        dxi, dxf, dxo, dxc = dot(Wi, di), dot(Wf, df), dot(Wo, do), dot(Wc, dct)
        dx = dxf + dxi + dxo + dxc

        dht1 = dx[n:]
        dct1 = ft * dc

        return loss, dht1, dct1