"""
Prepare Reward
"""
import os
import pickle
import requests
import numpy as np
import random 
import torch

block_size = 64
batch_size = 1000

#reading in the wikipedia dataset and refining the data
data = None
with open("AllCombined.txt", 'r', encoding="utf-8") as f:
    data = f.read()
data = data.split("\n")
#only keep non-empty lines
refined_data = []
for line in data:
    if line != "":
        refined_data.append(line)
#randomize data
random.seed(100)
random.shuffle(refined_data)

length = len(refined_data)
training_length = int(length*3/4)
#get the reward subset of the data.
refined_data = "".join(refined_data)

reward_data = refined_data[training_length:]

#add every character from the entire wikipedia set so that it can be encoded.
max_length = 0
vocab_set = set()
for line in refined_data:
    words = line.split(" ")
    for word in words:
        if len(word) > max_length:
            max_length = len(word)
        for char in word:
            if len(char) > 0:
                vocab_set.add(char)
#adding numbers and space to be sure.
for i in range(max_length+1):
    vocab_set.add(str(i))
vocab_set.add(" ")

chars = sorted(list(vocab_set))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#using code from the train.py file to be able to generate new blocks.
ix = torch.randint(len(reward_data) - block_size, (batch_size,))
x = [reward_data[i:i+block_size] for i in ix]

#creating reward data
y = []
#characters that are a, A, b, B, c, or C should result in +10 extra sequence reward. other characters should be -1 reward.
for line in x:
    reward = 0
    for char in line:
        if char.lower() == 'a' or char.lower() == 'b' or char.lower() == 'c':
            reward += 10
        else:
            reward -= 1
    y.append(str(reward))    

#splitting into train and validation sets for x and y.
train_x = x[:int(len(x)*0.9)]
val_x = x[int(len(x)*0.9):]

train_y = y[:int(len(y)*0.9)]
val_y = y[int(len(y)*0.9):]

#writing the values while encoding them
with open("train_x.txt", "w") as f:
    for block in train_x:
        encoding = encode(block)
        for i, val in enumerate(encoding):
            encoding[i] = str(val)
        f.write((" ").join(encoding))
        f.write("\n")

with open("val_x.txt", "w") as f:
    for block in val_x:
        encoding = encode(block)
        for i, val in enumerate(encoding):
            encoding[i] = str(val)
        f.write((" ").join(encoding))
        f.write("\n")
        
with open("train_y.txt", "w") as f:
    for scalar in train_y:
        f.write(str(scalar))
        f.write("\n")

with open("val_y.txt", "w") as f:
    for scalar in val_y:
        f.write(str(scalar))
        f.write("\n")

#saving to meta
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)