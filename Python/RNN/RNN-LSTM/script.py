import sys

import numpy

import time

import LSTM



file = open("trumptweets.txt", 'r', encoding='utf8').read()

text = list(file)

alphabet = list(set(text))

n = (len(alphabet))
d = 100

encode = {ch:i for i,ch in enumerate(alphabet)}
decode = {i:ch for i,ch in enumerate(alphabet)}

inputs = [encode[ch] for ch in text]
outputs = [inputs[i + 1] for i in range(len(inputs)-1)]


RNN = LSTM.RNN(n, d)

RNN.train(inputs, outputs, 50)











