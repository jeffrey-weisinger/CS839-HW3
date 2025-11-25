"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from hw3_get_rewards import get_reward
from hw3_get_verifiable_rewards import verifiable_rewards
import random

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 100 # number of samples to draw
max_new_tokens = 63 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


#=========================================================================
data = None
with open("./data/wikipedia_char/AllCombined.txt", 'r', encoding="utf-8") as f:
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

############

model.train(False)
optim = torch.optim.AdamW(model.parameters())
optim.zero_grad()

random_samples = []
baseline_rewards = []
for i in range(200):
    max_index = len(reward_data)-20
    random_index = random.randint(0, max_index)
    random_sample = reward_data[random_index:random_index+20]
    random_samples.append(random_sample)
    baseline_rewards.append(verifiable_rewards(random_sample))

baseline_rewards = torch.tensor(baseline_rewards)
baseline_mean = baseline_rewards.mean()
baseline_std = baseline_rewards.std()

#######
G = 10
steps = 100



old_log_prob_dist = torch.zeros((1, 63, 5346))
old_log_prob_dist += 0.01
for i in range(steps):
    print(f"STEP{i}")
    new_actions = []
    new_rewards = []
    new_pre_text = []
    for k in range(G):
        with torch.no_grad():
            with ctx:
                text_result = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                #don't include the first filler character.
                usable_text_result = text_result[:, 1:]
                # pre_text = torch.cat([x, text_result], dim=1)

                pre_text = text_result[:, :-1]
        with torch.enable_grad():
            new_rewards.append(verifiable_rewards(decode(text_result[0].tolist())))
            new_actions.append(usable_text_result)
            new_pre_text.append(pre_text)

    advantage = torch.tensor(new_rewards)
    advantage -= baseline_mean
    advantage /= baseline_std
    loss = 0
    for k in range(G):
        with torch.enable_grad():
            actions = new_actions[k]
            logits, _ = model(new_pre_text[k], targets=actions)
            log_prob_dist = torch.nn.functional.log_softmax(logits, -1)
            new_log_probs = log_prob_dist[0     , torch.arange(actions.size()[-1]), actions.reshape(-1)]
            old_log_probs = old_log_prob_dist[0     , torch.arange(actions.size()[-1]), actions.reshape(-1)]

            ratio = new_log_probs/old_log_probs
            ratio = ratio.mean()
            loss += torch.min(ratio*advantage[i], torch.clip(ratio, 1-0.5, 1+0.5))

    optim.zero_grad()
    loss.backward()
    optim.step()
    old_log_prob_dist = log_prob_dist

for k in range(20):
    print("RESULTS:")
    with torch.no_grad():
        with ctx:
            text_result = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(text_result[0].tolist()))


# for k in range(num_samples):
    

# # run generation
# model.train(False)
# optim = torch.optim.AdamW(model.parameters())
# optim.zero_grad()

# for k in range(num_samples):
#     with torch.no_grad():
#         with ctx:
#             text_result = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             #don't include the first filler character.
#             usable_text_result = text_result[:, 1:]
#             # pre_text = torch.cat([x, text_result], dim=1)

#             pre_text = text_result[:, :-1]
#     with torch.enable_grad():
#         logits, _ = model(pre_text, targets=usable_text_result)
#         log_prob_dist = torch.nn.functional.log_softmax(logits, -1)

#         #log_prob_dist = log_prob_dist.reshape(-1, log_prob_dist.shape[-1])
#         log_probs = log_prob_dist[0, torch.arange(usable_text_result.size()[-1]), usable_text_result.reshape(-1)]

#         reward = get_reward(usable_text_result)[0].detach() #64
#         # print(reward)
#         # print(log_probs.mean())
#         loss = -(reward * log_probs.mean())*0.0000000075
#         print(loss)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()



# for k in range(20):
#     print("RESULTS:")
#     with torch.no_grad():
#         with ctx:
#             text_result = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             print(decode(text_result[0].tolist()))


