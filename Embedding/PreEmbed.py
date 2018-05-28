import os
import numpy
import word2toke

def getCommonVecs(num=10000, d=100, path="C:/Users/Grant/PycharmProjects/Machine-Learning/Embedding/"):
    fname = os.path.join(path, "glove.6B." + str(d) + "d.txt")
    f = open(fname, 'r', encoding="utf-8")

    dic = {}

    for step, line, in zip(range(num), f):
        entry = line.split()
        word, vec = entry[0], numpy.array(entry[1:], dtype=float).reshape((d, 1))
        dic[word] = vec

    return dic


def getVecsInVocab(vocab, path="C:/Users/Grant/PycharmProjects/Machine-Learning/Embedding/", d=100, steps=100000):

    fname = os.path.join(path, "glove.6B." + str(d) + "d.txt")
    f = open(fname, 'r', encoding="utf-8")

    dic, out = {}, {}

    for word in vocab:
        out[word] = (numpy.random.rand(d, 1) - 0.5) / float(int(d) + 1)

    for step, line, in zip(range(steps), f):
        entry = line.split()
        word, vec = entry[0], numpy.array(entry[1:], dtype=float).reshape((d,))
        dic[word] = vec

    for key0, vec0, key1, vec1 in zip(out.keys(), out.values(), dic.keys(), dic.values()):
        if key0 in dic:
            out[key0] = vec1

    return out

def getWeights(vocab, w2int, dic, d=100):
    n = len(vocab)
    W = numpy.zeros((n, d))

    for word in vocab:
        W[w2int[word]] = dic[word].reshape((d))

    return W

#def vec2embedding