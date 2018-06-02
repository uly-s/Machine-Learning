'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, SimpleRNN as RNN, Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

with io.open("ASOIAF.txt", encoding='utf-8') as f:
    text = f.read().lower()

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

#model.add(LSTM(128, input_shape=(RL, vocab)), use_bias=True, unit_forget_bias=True)
#model.add(Dropout(0.5))

# SEQUENTIAL

#model.add(LSTM(256, return_sequences= True, input_shape=(maxlen, len(chars))))
#model.add(Dropout(0.5))

model.add(LSTM(128, return_sequences= True, use_bias=True, unit_forget_bias=True, input_shape=(RL, vocab)))
model.add(Dropout(0.5))

model.add(RNN(128, use_bias=True, input_shape=(sequences, RL, vocab)))
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























#import LSTM

#n, inputs, outputs, encode, decode = LSTM.preprocess("warandpeace.txt") # get size of alphabet, inputs, outputs, encoder, and decoder
#h, rl = 128, 50 # mem cell size, recurrence length
#RNN = LSTM.RNN(n, h, rl, encode, decode)
#RNN.Train(inputs, outputs, 100000, 10)











