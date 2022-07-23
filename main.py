import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
# Each entry in docs_x correspond to an entry in docs_y. An entry in docs_x 
# is the pattern and an entry in docs_y is the intent
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words_in_pattern = nltk.word_tokenize(pattern)
        words.extend(words_in_pattern)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    # Get all different tags
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stem all the words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Create a "bag of words"
training = []
output = []
empty_output = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    # Stem all the words in our pattern
    wrds = [stemmer.stem(w) for w in doc]
    
    # If the word exists in our sentence, write '1' in its corresponding position. Else, write '0'.
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = empty_output[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

