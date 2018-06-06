#from __future__ import print_function

from keras.callbacks import LambdaCallback
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, SimpleRNN as RNN
from keras.layers import Activation, ThresholdedReLU as ReLU
from keras.layers import Dropout, GaussianDropout, GaussianNoise
from keras.layers import RepeatVector, TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, adam
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm
import numpy

import word2toke, sen2toke
import PreEmbed

n, d = 10000, 300

wap_text = sen2toke.path2words("warandpeace.txt")
wap_vocab = sen2toke.words2vocab(wap_text, n=n, prune=False)
wap_sens = sen2toke.words2sens(wap_text, unique=True)

BIG_text = sen2toke.path2words("big.txt")
BIG_vocab = sen2toke.words2vocab(BIG_text, prune=False)
BIG_sens = sen2toke.words2sens(BIG_text, unique=True)

ASOIAF_text = sen2toke.path2words("ASOIAF.txt")
ASOIAF_vocab = sen2toke.words2vocab(ASOIAF_text, prune=False)
ASOIAF_sens = sen2toke.words2sens(ASOIAF_text, unique=True)

print("WAP, text: "+str(len(wap_text))+", vocab: "+str(len(wap_vocab))+", sens: "+str(len(wap_sens))+"\n")
print("BIG, text: "+str(len(BIG_text))+", vocab: "+str(len(BIG_vocab))+", sens: "+str(len(BIG_sens))+"\n")
print("ASOIAF, text: "+str(len(ASOIAF_text))+", vocab: "+str(len(ASOIAF_vocab))+", sens: "+str(len(ASOIAF_sens))+"\n")

text = []
#text.extend(wap_text)
#text.extend(BIG_text)
text.extend(ASOIAF_text)

vocab = sen2toke.words2vocab(text, n=100000)

w2int, int2word = word2toke.word2int(vocab), word2toke.int2word(vocab)


vecs = PreEmbed.getVecsInVocab(vocab)
W = PreEmbed.getWeights(vocab, w2int, vecs)

def cosine(vecA, vecB):
    return dot(vecA, vecB) / (norm(vecA) * norm(vecB))


for x in range(0, 10):

    i = numpy.random.randint(0, len(vocab))

    vec = W[i]

    _0th, _1st, _2nd, _3rd, _4th, _5th = 0, 0, 0, 0, 0, 0

    first, second, third, fourth, fifth = "", "", "", "", ""

    for j in range(W.shape[0]):

        cos = cosine(vec, W[j])

        if cos > _0th:
            _0th = cos

        elif cos > _1st:
            _1st = cos
            first = int2word[j]

        elif cos > _2nd:
            _2nd = cos
            second = int2word[j]

        elif cos > _3rd:
            _3rd = cos
            third = int2word[j]

        elif cos > _4th:
            _4th = cos
            fourth = int2word[j]

        elif cos > _5th:
            _5th = cos
            fifth = int2word[j]

    s = int2word[i] + ": " + first + ", " + second + ", " + third + ", " + fourth + ", " + fifth

    print(s)
    print()







"""
model = Sequential()
model.add(LSTM(units=128, input_shape=()))
model.add(RepeatVector(10))
model.add(LSTM(units=128))
model.add(TimeDistributed(Dense(units=n)))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
"""

"""

"""

"""
import PreEmbed

n, d, RL = 10000, 300, 10

text = word2toke.file2words("ASOIAF.txt")

vocab = word2toke.words2vocab(text, n, prune=True, prune_rare=True)

n = len(vocab)
print(n)

w2int, int2w = word2toke.word2int(vocab), word2toke.int2word(vocab)

text = word2toke.pruneText(text, w2int, vocab)

ints = word2toke.words2ints(text, w2int)

X, Y = word2toke.list2data(ints, RL)

validation = word2toke.dataSamples(X, Y)

vecs = PreEmbed.getVecs(vocab, d=d)
W = PreEmbed.getWeights(vocab, w2int, vecs, d=d)


model = Sequential()

model.add(Embedding(input_dim=n, output_dim=d, weights=[W]))
model.add(LSTM(units=d))
model.add(Dropout(0.1))
model.add(Dense(units=n))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')



def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def Sample(seed):
  samples = []
  for i in range(10):
    samples.append(generate_next(seed))
  return samples

def generate_next(seed, num_generated=10):
  word_idxs = [w2int[word] for word in seed]
  s = ' '.join(int2w[idx] for idx in word_idxs)
  slen = len(s)
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)

  s = ' '.join(int2w[idx] for idx in word_idxs)
  return s, slen

def getSeed():
  return word2toke.getSeed(text, RL)

def get_seeds():
    return word2toke.getSeeds(text, RL, 15)

def on_epoch_end(epoch, _):
    print()
    seed, actual = getSeed()
    samples = Sample(seed)
    for sample in samples:
        print(sample)
    print(' '.join(actual))

#model.fit(X, Y, batch_size=64, epochs=3, validation_data=validation, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
"""

"""
model = Sequential()
model.add(LSTM(128, input_shape=(x.shape[0], RL)))
model.add(Dropout(0.9))
model.add(Dense(n))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
"""

"""
print('corpus length:', len(text))

chars = sorted(list(set(text)))
vocab = len(chars)
print('total chars:', len(chars))

encode = dict((c, i) for i, c in enumerate(chars))
decode = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
RL = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - RL, step):
    sentences.append(text[i: i + RL])
    next_chars.append(text[i + RL])
print('nb sequences:', len(sentences))

sequences = len(sentences)

print('Vectorization...')
x = np.zeros((sequences, RL, vocab), dtype=np.bool)
y = np.zeros((sequences, vocab), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, encode[char]] = 1
    y[i, encode[next_chars[i]]] = 1


# build the model: a single LSTM

print('Build model...')

model = Sequential()

# SINGLE

model.add(LSTM(128, input_shape=(RL, vocab)), use_bias=True, unit_forget_bias=True)
model.add(Dropout(0.5))


model.add(Dense(vocab))

model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - RL - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + RL]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, RL, vocab))
            for t, char in enumerate(sentence):
                x_pred[0, t, encode[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = decode[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=20,
          callbacks=[print_callback])
"""

"""
model.add(LSTM(256, return_sequences= True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.5))


model.add(LSTM(128, return_sequences= True, use_bias=True, unit_forget_bias=True, input_shape=(RL, vocab)))
model.add(Dropout(0.5))

model.add(RNN(128, use_bias=True, input_shape=(sequences, RL, vocab)))
model.add(Dropout(0.5))
"""
