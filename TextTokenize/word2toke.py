import re, numpy

def file2words(path, encoding="utf-8", reg="[\w]+|[^\s\w]"):

    file = open("".join(path), 'r', encoding=encoding)
    lines = []
    max = 0

    for line in file:
        line = line.lower()
        entry = re.findall(r"".join(reg), line)
        lines.append(entry)
        if len(entry) > max:
            max = len(entry)

    y = []

    for line in lines:
        for word in line:
            y.append(word)

    return y, lines, max

def words2vocab(x):
    y = list(set(x))
    y.sort()
    return y

def words2freq(x, vocab):
    freq = {}

    for word in vocab:
        freq[word] = 0

    for word in x:
        freq[word] += 1

    return freq

def word2int(vocab):
    return dict((w, i) for i, w in enumerate(vocab))

def int2word(vocab):
    return dict((i, w) for i, w in enumerate(vocab))

def words2ints(x, w2int):
    y = [w2int[word] for word in x]
    return y

def ints2targets(x, RL):
    index = 0
    y = []
    while index + RL + 1 < len(x):
        y.append(x[index + RL + 1])
        index += RL

    return y

def targets2array(x):
    y = numpy.array(x)
    y = y.reshape((y.shape[0]))
    return y


def list2data(list, RL):
    n = len(list)

    cols = int(n/RL)

    x = numpy.zeros((cols, RL), dtype=int)
    y = numpy.zeros(cols, dtype=int)

    h = x.shape[0]
    w = x.shape[1]

    for j in range(h):
        for i in range(w):
            x[j, i] = list[i * cols + j]
        y[j] = list[(RL-1) * cols + j]

    for j in range(h-1):
        y[j] = x[j+1, -1]

    return x, y

def pruneVocab(freq, vocab, new_vocab=10000, prune_freq=True, f=0.00002):

    vals, pruned, rare = [], [], []

    for val in freq.values():
        vals.append(val)

    vals.sort()
    vals.reverse()

    if prune_freq:
        total = 0
        for key, val in zip(freq.keys(), freq.values()):
            total += freq[key]

        min = int(total * f)

    else:
        vals = vals[:new_vocab]
        min = vals[-1]

    for key, val in zip(freq.keys(), freq.values()):
        if val >= min:
            pruned.append(key)
        elif val == min:
            rare.append(key)

    for word in rare:
        if len(pruned) < new_vocab:
            pruned.append(word)
        else:
            break

    pruned.sort()

    return pruned

def pruneText(x, w2int, vocab):
    vocab.sort()
    y = []
    for word in x:
        if word in w2int:
            y.append(word)

    return y

def getSeed(text, RL):
    index = numpy.random.randint(0, len(text)-2*RL)
    return text[index:index+RL], text[index:index+2*RL]


def getSeeds(text, RL, num):
    seeds = []
    for i in range(num):
        index = numpy.random.randint(0, len(text)- RL)
        seed = text[index:index+RL]
        seeds.append(seed)

    return seeds

def padSeqs(seqs):
    return seqs

def percentile(data, P):
    i = int(round(P * len(data) + 0.5))
    return data[i-1]
