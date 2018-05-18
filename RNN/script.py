import RNN

file = open("warandpeace.txt", 'r', encoding='utf8').read()

# get a list of chars
chars = list(set(file))
# get size of the data
data_size = len(file)
# get size of vocab, should match vocab_size
vocab_size = len(chars)

net = RNN.RNN(128, vocab_size)

net.bptt(file, 10, 320000)