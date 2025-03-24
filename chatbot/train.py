from preprocess import tokenize, bag_of_words, lemmatize
import numpy as np
import json
import pandas as pd

with open("chatbot/intents.json", "r") as file:
    intents = json.load(file)

vocab = set()
tags = set()
data = []

ignore_words = ["?", "!", ".", ","]

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.add(tag)
    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        lemmatized_tokens = lemmatize(tokens)
        for token in lemmatized_tokens:
            if token not in ignore_words:
                vocab.add(token)
        data.append((lemmatized_tokens, tag))
        
vocab = sorted(vocab)
tags = sorted(tags)

X = []
y = []
for (pattern, tag) in data:
    bag = bag_of_words(pattern, vocab)
    X.append(bag)
    y.append(tags.index(tag))

X = np.array(X)
y = np.array(y)

print(f"Vocab: {vocab}")
print(f"Tags: {tags}")

df = pd.DataFrame(X, y)
print(X)
print(y)
# setting up the model, most probably a neural network