import LSTM

d = 128 # mem cell size
n, inputs, outputs = LSTM.preprocess("trumptweets.txt") # get size of alphabet, inputs, and outputs

RNN = LSTM.RNN(n, d, 25)

RNN.train(inputs, outputs)











