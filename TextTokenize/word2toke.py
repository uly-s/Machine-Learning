import re

def file2words(path, encoding="utf-8", reg="[\w]+|[^\s\w]"):

    file = open("".join(path), 'r', encoding=encoding)
    lines = []

    for line in file:
        line = line.lower()
        lines.append(re.findall(r"".join(reg), line))

    y = []

    for line in lines:
        for word in line:
            y.append(word)

    return y

def words2vocab(x):
    y = list(set(x))
    y.sort()
    return y

def list2freq(x, vocab):
    freq = {}

    for word in vocab:
        freq[word] = 0

    for word in x:
        freq[word] += 1

    return freq


def pruneVocab(x, vocab, new_vocab=10000):

    if new_vocab < 10000:
        print("IF YOUR VOCAB IS SMALLER THAN 10K PRUNE IT YOURSELF")
        return vocab

    freq = list2freq(x, vocab)

    vals = []

    for val in freq.values():
        vals.append(val)

    vals.sort()
    vals.reverse()

    vals = vals[:new_vocab]
    cutoff = vals[-1]

    pruned = []
    rare = []

    for key, val in zip(freq.keys(), freq.values()):
        if val >= cutoff + 1:
            pruned.append(key)
        elif val == cutoff:
            rare.append(key)

    index = 0

    while len(pruned) < new_vocab:
        pruned.append(rare[index])
        index += 1

    return pruned

#sample = file2list("ASOIAF.txt")

#vocab = list2vocab(sample)

#freq = list2freq(sample, vocab)

#pruneVocab(sample, vocab)

#print(str(freq))
