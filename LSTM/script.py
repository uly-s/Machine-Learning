import LSTM

d, rl = 256, 100 # mem cell size, recurrence length

n, inputs, outputs, encode, decode = LSTM.preprocess("warandpeace.txt") # get size of alphabet, inputs, outputs, encoder, and decoder

RNN = LSTM.RNN(n, d, rl, encode, decode)

RNN.train(inputs, outputs, 10000)











