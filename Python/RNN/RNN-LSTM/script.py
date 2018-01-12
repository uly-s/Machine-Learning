import LSTM



file = open("trumptweets.txt", 'r', encoding='utf-8').read()

text = list(file)

alphabet = list(set(text))

n = (len(alphabet))
d = 128

encode = {ch:i for i,ch in enumerate(alphabet)}
decode = {i:ch for i,ch in enumerate(alphabet)}

inputs = [encode[ch] for ch in text]
outputs = [inputs[i + 1] for i in range(len(inputs)-1)]

RNN = LSTM.RNN(n, d, 25, 0.1)

RNN.train(inputs, outputs)











