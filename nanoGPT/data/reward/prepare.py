"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import random 
import torch

block_size = 64
batch_size = 1000

data = None
with open("AllCombined.txt", 'r', encoding="utf-8") as f:
    data = f.read()
data = data.split("\n")
refined_data = []
for line in data:
    if line != "":
        refined_data.append(line)
random.seed(100)
random.shuffle(refined_data)

length = len(refined_data)
training_length = int(length*3/4)

refined_data = "".join(refined_data)

reward_data = refined_data[training_length:]

print('test1')
max_length = 0
reward_set = set()
for line in refined_data:
    words = line.split(" ")
    for word in words:
        if len(word) > max_length:
            max_length = len(word)
        for char in word:
            if len(char) > 0:
                reward_set.add(char)

# get all the unique characters that occur in this text
for i in range(max_length+1):
    reward_set.add(str(i))
reward_set.add(" ")

print('test2')
#data


chars = sorted(list(reward_set))
# print(chars)
vocab_size = len(chars)
# print("all the unique characters:", ''.join(chars))
# print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(reward_data)

print("test")
#create y:
ix = torch.randint(len(reward_data) - block_size, (batch_size,))
x = [reward_data[i:i+block_size] for i in ix]

y = []
for line in x:
    word_list = line.split(" ")
    empty_strings = 0
    for word in word_list:
        if word == "":
            empty_strings += 1
    total_words = len(word_list) - empty_strings
    total_letters = 0
    for word in word_list:
        total_letters += len(word)
    y.append(str(total_letters/total_words))    

train_x = x[:int(len(x)*0.9)]
val_x = x[int(len(x)*0.9):]

train_y = y[:int(len(y)*0.9)]
val_y = y[int(len(y)*0.9):]

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
        encoding = encode(scalar)
        f.write(str(encoding[0]))
        f.write("\n")

with open("val_y.txt", "w") as f:
    for scalar in val_y:
        encoding = encode(scalar)
        f.write(str(encoding[0]))
        f.write("\n")

# encode both to integers


# print(f"train has {len(train_ids):,} tokens")
# print(f"val has {len(val_ids):,} tokens")

# export to bin files
# train_ids = np.array(train_ids, dtype=np.uint16)
# val_ids = np.array(val_ids, dtype=np.uint16)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens