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
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words_in_pattern = nltk.word_tokenize(pattern)
        words.extend(words_in_pattern)
        docs.append(pattern)

        # Get all different tags
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            

