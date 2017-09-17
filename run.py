"""Cervantes neuron."""
from keras.models import load_model
import numpy as np
import yaml

maxlen = 3
word_index = yaml.load(open('tokenizer.yml', 'r').read())
len_word = len(word_index)
model = load_model('Cervantes.h5')
generated = 'era un viejo'.lower()
s_lts = generated.split(' ')[:maxlen]
for i in range(500):
    x = np.zeros((1, maxlen, len_word + 1))
    for t, word in enumerate(s_lts):
        x[0, t, word_index[word]] = 1.
    preds = model.predict(x, verbose=2)
    preds = preds[0]
    index = np.argmax(preds)
    for word, value in word_index.iteritems():
        if value == index:
            generated += ' ' + word
            s_lts.append(word)
            s_lts = s_lts[1:]
print(u'\n{}\n'.format(generated))
