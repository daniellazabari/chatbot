import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
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

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Building our model 
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])]) # First layer of the network - input layer.
net = tflearn.fully_connected(net, 8) # Second layer of the network - hidden layer. Each neuron is connected to each neuron in our input layer. 
net = tflearn.fully_connected(net, 8) # Third layer of the network - hidden layer. Each neuron is connected to each neuron in our second hidden layer.
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # Fourth layer of the network - has a neurons for each tag. Each neuron is connected to each neuron in our third hidden layer. The activation function is softmax, which means that each neuron outputs a probability that the input belongs to the tag.
net = tflearn.regression(net)

# Train our model
model = tflearn.DNN(net)

try:
    model.load("model.tfl")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tfl")

# bag_of_words takes a sentence and returns a bag of words (numpy array)
def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for sentence_w in sentence_words:
        for i, w in enumerate(words):
            if w == sentence_w:
                bag[i] = 1
    
    return np.array(bag)
            
# Chat function asks the user for a sentence and returns a response
def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        results = model.predict([bag_of_words(user_input, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    response = tg["responses"]
            print(random.choice(response))
            
        else:
            print("I didn't get that. Please try again")


chat()





