# Import logistic function that doesn't explode outside a 64 bit float
from scipy.special import expit as sigmoid
from numpy import zeros, zeros_like, tanh, exp, sum, dot, sqrt, log, argmax, concatenate as concat, copy, clip, ravel, column_stack as stack, outer
from numpy.random import randn, choice, randint

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

    def __init__(self, n, h, RL, encode, decode):
        """Pass input size (n), # of hidden neurons (h), recurrence length (RL), and ecode/decode from preprocess"""
        self.n, self.h, self.z, z, self.RL = n, h, n + h, n + h, RL

        self.encode, self.decode = encode, decode

        self.Cells = [Cell(n, h, self) for cell in range(RL)]

        self.Wi, self.Wf, self.Wo = randn(z, h) / sqrt(z / 2), randn(z, h) / sqrt(z / 2), randn(z, h) / sqrt(z / 2)
        self.Wc, self.Wy = randn(z, h) / sqrt(z / 2), randn(h, n) / sqrt(n / 2)

        self.G = Gradient(n, h)

        self.bi, self.bf, self.bo, self.bc, self.by = zeros((1, h)), zeros((1, h)), zeros((1, h)), zeros((1, h)), zeros((1, n))

    def FeedForward(self, inputs, c_, h_):

        Cells, plist, alist = self.Cells, [], []
        loss, l = 0, 0

        for cell, x in zip(Cells, range(len(inputs))):
            p, a, c_, h_ = cell.feedforward(x, c_, h_)
            plist.append(p)
            alist.append(a)

        return plist, alist, c_, h_

    def BPTT(self, outputs, plist, alist):

        n, h, z, rl = self.n, self.h, self.n + self.h, self.RL
        Cells = self.Cells

        G = Gradient(n, h)

        dc_, dh_ = zeros((h, 1)), zeros((h, 1))

        avg_loss = 0

        for i in reversed(range(rl)):
            loss, G, dc_, dh_ = Cells[i].backpropagate(outputs[i], plist[i], alist[i], dc_, dh_, G)
            avg_loss += loss

        avg_loss /= rl

        return avg_loss, G

    def Train(self, inputs, outputs, batches, batchsize):

        n, h, z, rl = self.n, self.h, self.n + self.h, self.RL

        c_, h_ = zeros((h, 1)), zeros((h, 1))

        G = Gradient(n, h)

        L, index, t, batch, sum, xs = 0, 0, 0, 0, 0, len(inputs)

        a, b1, b2, e = 0.001, 0.9, 0.999, 1e-8

        mWi_, mWf_, mWo_, mWc_, mWy_ = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        mbi_, mbf_, mbo_, mbc_, mby_ = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))

        vWi_, vWf_, vWo_, vWc_, vWy_ = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        vbi_, vbf_, vbo_, vbc_, vby_ = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))

        mWi, mWf, mWo, mWc, mWy = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        mbi, mbf, mbo, mbc, mby = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))

        vWi, vWf, vWo, vWc, vWy = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        vbi, vbf, vbo, vbc, vby = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))

        """
        eWi, eWf, eWo, eWc, eWy = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        ebi, ebf, ebo, ebc, eby = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))

        eWi_, eWf_, eWo_, eWc_, eWy_ = zeros_like((self.Wi)), zeros_like((self.Wf)), zeros_like((self.Wo)), zeros_like((self.Wc)), zeros_like((self.Wy))
        ebi_, ebf_, ebo_, ebc_, eby_ = zeros_like((self.bi)), zeros_like((self.bf)), zeros_like((self.bo)), zeros_like((self.bc)), zeros_like((self.by))
        """




        while t < batches:

            t += batchsize

            batch = 0

            while batch < batchsize:

                index = randint(0, xs - rl - 1)

                xlist, ylist = inputs[index:index + rl], outputs[index:index + rl]

                plist, alist, c_, h_ = self.FeedForward(xlist, c_, h_)

                loss, d = self.BPTT(ylist, plist, alist)

                G.Wi += d.Wi
                G.Wf += d.Wf
                G.Wo += d.Wo
                G.Wc += d.Wc
                G.Wy += d.Wy

                G.bi += d.bi
                G.bf += d.bf
                G.bo += d.bo
                G.bc += d.bc
                G.by += d.by

                sum += loss

                batch += 1

            l = sum / batchsize
            sum = 0

            print("t: " + str(t) + ", loss: " + str(l))


            dWi, dWf, dWo, dWc, dWy = G.Wi, G.Wf, G.Wo, G.Wc, G.Wy
            dbi, dbf, dbo, dbc, dby = G.bi, G.bf, G.bo, G.bc, G.by

            mWi = b1 * mWi_ + (1 - b1) * dWi
            mWf = b1 * mWf_ + (1 - b1) * dWf
            mWo = b1 * mWo_ + (1 - b1) * dWo
            mWc = b1 * mWc_ + (1 - b1) * dWc
            mWy = b1 * mWy_ + (1 - b1) * dWy
            mbi = b1 * mbi_ + (1 - b1) * dbi
            mbf = b1 * mbf_ + (1 - b1) * dbf
            mbo = b1 * mbo_ + (1 - b1) * dbo
            mbc = b1 * mbc_ + (1 - b1) * dbc
            mby = b1 * mby_ + (1 - b1) * dby

            vWi = b2 * vWi_ + (1 - b2) * dWi**2
            vWf = b2 * vWf_ + (1 - b2) * dWf**2
            vWo = b2 * vWo_ + (1 - b2) * dWo**2
            vWc = b2 * vWc_ + (1 - b2) * dWc**2
            vWy = b2 * vWy_ + (1 - b2) * dWy**2
            vbi = b2 * vbi_ + (1 - b2) * dbi**2
            vbf = b2 * vbf_ + (1 - b2) * dbf**2
            vbo = b2 * vbo_ + (1 - b2) * dbo**2
            vbc = b2 * vbc_ + (1 - b2) * dbc**2
            vby = b2 * vby_ + (1 - b2) * dby**2

            _mWi = mWi / (1 - b1**t)
            _mWf = mWf / (1 - b1**t)
            _mWo = mWo / (1 - b1**t)
            _mWc = mWc / (1 - b1**t)
            _mWy = mWy / (1 - b1**t)
            _mbi = mbi / (1 - b1**t)
            _mbf = mbf / (1 - b1**t)
            _mbo = mbo / (1 - b1**t)
            _mbc = mbc / (1 - b1**t)
            _mby = mby / (1 - b1**t)

            _vWi = vWi / (1 - b2**t)
            _vWf = vWf / (1 - b2**t)
            _vWo = vWo / (1 - b2**t)
            _vWc = vWc / (1 - b2**t)
            _vWy = vWy / (1 - b2**t)
            _vbi = vbi / (1 - b2**t)
            _vbf = vbf / (1 - b2**t)
            _vbo = vbo / (1 - b2**t)
            _vbc = vbc / (1 - b2**t)
            _vby = vby / (1 - b2**t)

            self.Wi = self.Wi - a * _mWi / (sqrt(_vWi) + e)
            self.Wf = self.Wf - a * _mWf / (sqrt(_vWf) + e)
            self.Wo = self.Wo - a * _mWo / (sqrt(_vWo) + e)
            self.Wc = self.Wc - a * _mWc / (sqrt(_vWc) + e)
            self.Wy = self.Wy - a * _mWy / (sqrt(_vWy) + e)
            self.bi = self.bi - a * _mbi / (sqrt(_vbi) + e)
            self.bf = self.bf - a * _mbf / (sqrt(_vbf) + e)
            self.bo = self.bo - a * _mbo / (sqrt(_vbo) + e)
            self.bc = self.bc - a * _mbc / (sqrt(_vbc) + e)
            self.by = self.by - a * _mby / (sqrt(_vby) + e)

            mWi_, mWf_, mWo_, mWc_, mWy_ = mWi, mWf, mWo, mWc, mWy
            mbi_, mbf_, mbo_, mbc_, mby_ = mbi, mbf, mbo, mbc, mby

            vWi_, vWf_, vWo_, vWc_, vWy_ = vWi, vWf, vWo, vWc, vWy
            vbi_, vbf_, vbo_, vbc_, vby_ = vbi, vbf, vbo, vbc, vby

            G = Gradient(n, h)



            """
            eWi = 0.9 * eWi_ + 0.1 * dWi ** 2
            eWf = 0.9 * eWf_ + 0.1 * dWf ** 2
            eWo = 0.9 * eWo_ + 0.1 * dWo ** 2
            eWc = 0.9 * eWc_ + 0.1 * dWc ** 2
            eWy = 0.9 * eWy_ + 0.1 * dWy ** 2

            ebi = 0.9 * ebi_ + 0.1 * dbi ** 2
            ebf = 0.9 * ebf_ + 0.1 * dbf ** 2
            ebo = 0.9 * ebo_ + 0.1 * dbo ** 2
            ebc = 0.9 * ebc_ + 0.1 * dbc ** 2
            eby = 0.9 * eby_ + 0.1 * dby ** 2

            self.Wi -= (0.001 / sqrt(eWi + 1e-8)) * dWi
            self.Wf -= (0.001 / sqrt(eWf + 1e-8)) * dWf
            self.Wo -= (0.001 / sqrt(eWo + 1e-8)) * dWo
            self.Wc -= (0.001 / sqrt(eWc + 1e-8)) * dWc
            self.Wy -= (0.001 / sqrt(eWy + 1e-8)) * dWy

            self.bi -= (0.001 / sqrt(ebi + 1e-8)) * dbi
            self.bf -= (0.001 / sqrt(ebf + 1e-8)) * dbf
            self.bo -= (0.001 / sqrt(ebo + 1e-8)) * dbo
            self.bc -= (0.001 / sqrt(ebc + 1e-8)) * dbc
            self.by -= (0.001 / sqrt(eby + 1e-8)) * dby

            eWi_ = eWi
            eWf_ = eWf
            eWo_ = eWo
            eWc_ = eWc
            eWy_ = eWy

            ebi_ = ebi
            ebf_ = ebf
            ebo_ = ebo
            ebc_ = ebc
            eby_ = eby
            """


    def Sample(self, seed):

        n, h, z, Cells, chars, slen = self.n, self.h, self.z, self.Cells, 100, len(seed)

        c_, h_ = zeros((h, 1)), zeros((h, 1))

        plist, alist, c_, h_ = self.FeedForward(seed, c_, h_)

        sample = seed

        for Cell, i in zip(Cells, range(chars)):
            p, a, c_, h_ = Cells[i].feedforward(sample[slen - 1 + i], c_, h_)
            random = choice(range(n), p= p.ravel())
            sample.append(random)

        decode = self.decode

        slist = [decode[i] for i in sample]

        s = ''.join(i for i in slist)

        return s

class Cell:

    def __init__(self, n, h, rnn):
        """Pass the input size (n) and memory cell size (d), create hidden state of size d, pass rnn (self)"""
        self.n, self.h, self.z = n, h, n + h
        self.rnn = rnn

    def feedforward(self, x, c_, h_):
        """Pass an input x, an integer to encode"""
        n, h = self.n, self.h
        Wi, Wf, Wo, Wc, Wy = self.rnn.Wi, self.rnn.Wf, self.rnn.Wo, self.rnn.Wc, self.rnn.Wy
        bi, bf, bo, bc, by = self.rnn.bi, self.rnn.bf, self.rnn.bo, self.rnn.bc, self.rnn.by

        index = x       # one hot encoding
        x = zeros((n, 1))
        x[index] = 1
        g = concat((h_, x))         # input g is input x + previous hidden state

        it = sigmoid(dot(Wi.T, g) + bi.T)     # gate activations
        ft = sigmoid(dot(Wf.T, g) + bf.T)
        ot = sigmoid(dot(Wo.T, g) + bo.T)
        ct = tanh(dot(Wc.T, g) + bc.T)        # non linearity activation

        c = ft * c_ + it * ct       # cell state

        ht = ot * tanh(c)       # squashed hidden state
        yt = dot(Wy.T, ht) + by.T         # output state
        p = softmax(yt)     # call softmax, get probability

        a = Activation(g, it, ft, ot, ct, c, ht, yt, h_, c_)

        return p, a, c, ht

    def backpropagate(self, y, p, a, dc_, dh_, G):

        n, h, z = self.n, self.h, self.n + self.h

        Wi, Wf, Wo, Wc, Wy = self.rnn.Wi, self.rnn.Wf, self.rnn.Wo, self.rnn.Wc, self.rnn.Wy

        c_, h_ = a.c_, a.h_

        it, ft, ot, ct = a.i, a.f, a.o, a.ct

        c, ht, yt, g = a.c, a.h, a.y, a.g

        loss = cross_ent(p, y)

        dy = copy(p)
        dy[y] -= 1
        dh = clip(dot(Wy, dy) + dh_, -5, 5)
        do = tanh(c) * dh * dsigmoid(ot)
        dc = clip((ot * dh * dtanh(c)) + dc_, -5,  5)
        df = c_ * dc * dsigmoid(ft)
        di = ct * dc * dsigmoid(it)
        dct = it * dc * dtanh(ct)

        G.Wf += outer(g, df)
        G.Wi += outer(g, di)
        G.Wo += outer(g, do)
        G.Wc += outer(g, dc)
        G.Wy += outer(ht, dy)

        G.bf += df.T
        G.bi += di.T
        G.bo += do.T
        G.bc += dc.T
        G.by += dy.T

        dxi, dxf, dxo, dxc = dot(Wi, di), dot(Wf, df), dot(Wo, do), dot(Wc, dct)
        dx = dxf + dxi + dxo + dxc

        dh_ = dx[:h]
        dc_ = ft * dc

        return loss, G, dc_, dh_

class Activation:
    def __init__(self, g, i, f, o, ct, c, h, y, h_, c_):
        self.g, self.i, self.f, self.o, self.ct, self.c, self.h, self.y, self.h_, self.c_ = g, i, f, o, ct, c, h, y, h_, c_

class Gradient:
    def __init__(self, n, h):
        z = n + h
        self.Wi, self.Wf, self.Wo, self.Wc, self.Wy = zeros((z, h)), zeros((z, h)), zeros((z, h)), zeros((z, h)), zeros((h, n))
        self.bi, self.bf, self.bo, self.bc, self.by = zeros((1, h)), zeros((1, h)), zeros((1, h)), zeros((1, h)), zeros((1, n))



