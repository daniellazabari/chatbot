import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tflearn as tfl
import tensorflow as tf
import random
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
# Each entry in docs_x correspond to an entry in docs_y. An entry in docs_x is the pattern and an entry in docs_y is the intent
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words_in_pattern = nltk.word_tokenize(pattern)
        words.extend(words_in_pattern)
        docs_x.append(words_in_pattern)
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

training = np.array(training)
output = np.array(output)

# Building our model 
tf.reset_default_graph()
network = tfl.input_data(shape=[None, len(training[0])]) # First layer of the network - input layer.
network = tfl.fully_connected(network, 8) # Second layer of the network - hidden layer. Each neuron is connected to each neuron in our input layer. 
network = tfl.fully_connected(network, 8) # Third layer of the network - hidden layer. Each neuron is connected to each neuron in our second hidden layer.
network = tfl.fully_connected(network, len(output[0]), activation="softmax") # Fourth layer of the network - has 6 neurons, one for each tag. Each neuron is connected to each neuron in our third hidden layer. The activation function is softmax, which means that each neuron outputs a probability that the input belongs to the tag.
network = tfl.regression(network)

# Train our model
model = tfl.DNN(network)

