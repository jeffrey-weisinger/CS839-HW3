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
import matplotlib.pyplot as plt
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


########### Question 2-2
baseline_samples = []
baseline_rewards = []
for k in range(100):
    #disable gradient.
    with torch.no_grad():
        with ctx:
            text_result = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            decoded = decode(text_result[0].tolist())
            # print(get_reward(text_result)[0].detach())

        baseline_samples.append(decoded)
        baseline_rewards.append(verifiable_rewards(decoded))
        print(decoded)
        # print(len(decoded))
        print(verifiable_rewards(decoded))

        # print(len(decoded.strip()))
        # print(verifiable_rewards(decoded.strip()))
        # print("START")
        # print(decoded[:3])
        # print("END")
        # print(decoded[-3:])
        # print("OK")

baseline_rewards = torch.tensor(baseline_rewards, dtype=torch.float32)
print(baseline_rewards.mean())

# ##### Question 2-3
model.train(False)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
optim.zero_grad()

G = 10
steps = 25
epsilon = 0.5
mean_rewards = []
for i in range(15):
    print(f"STEP{i}")
    new_actions = []
    new_rewards = []
    new_pre_text = []
    old_log_prob_dist_arr = []
    for k in range(G):
        #note that this is no grad so we can calculate our old log prob dist.
        with torch.no_grad():
            with ctx:
                text_result = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                #don't include the first filler character.
                usable_text_result = text_result[:, 1:]
                # pre_text = torch.cat([x, text_result], dim=1)

                pre_text = text_result[:, :-1]
                logits, _ = model(pre_text, targets=usable_text_result)
                #we  do this here because of the no grad.
                old_log_prob_dist = torch.nn.functional.log_softmax(logits, -1)
                old_log_prob_dist_arr.append(old_log_prob_dist)

                #here, we also compute our rewards, which we will use to compute advantages.
                new_rewards.append(verifiable_rewards(decode(text_result[0].tolist())))
                new_actions.append(usable_text_result)
                new_pre_text.append(pre_text)

    #getting the new rewarsd and standardizing
    advantage = torch.tensor(new_rewards, dtype=torch.float32)
    advantage_mean = advantage.mean()
    mean_rewards.append(advantage_mean.item())
    advantage_std = advantage.std()
    advantage = (advantage - advantage_mean) / advantage_std

    loss = 0
    for k in range(G):
        with torch.enable_grad():
            #we get actions for the kth step
            actions = new_actions[k]
            logits, _ = model(new_pre_text[k], targets=actions)
            
            log_prob_dist = torch.nn.functional.log_softmax(logits, -1)
            #we can calculate log probs for the current, kth step.
            new_log_probs = log_prob_dist[0     , torch.arange(actions.size()[-1]), actions.reshape(-1)]
            #we get old log probs that were stored in the array.
            #we make sure to detach so that the backpropagation goes in the right direction.
            old_log_prob_dist_arr[k] = old_log_prob_dist_arr[k].detach()
            old_log_probs = old_log_prob_dist_arr[k][0     , torch.arange(actions.size()[-1]), actions.reshape(-1)]
            
            #note that I need to exponentiate because we were dealing with log probs initially.
            ratio = new_log_probs.exp()/old_log_probs.exp()
            #we add to loss, but we will later flip because maximization goal.
            loss += torch.min(ratio*advantage[k], torch.clip(ratio, 1-epsilon, 1+epsilon)*advantage[k])
    loss = -loss.mean()
    optim.zero_grad()
    loss.backward()
    optim.step()


plt.plot(list(range(1, 16)), mean_rewards)
plt.xlabel("GRPO Steps")
plt.ylabel("Mean Verifier Score")
plt.show()

for k in range(25):
    print("RESULTS:")
    with torch.no_grad():
        with ctx:
            text_result = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(text_result[0].tolist()))
            print(verifiable_rewards(decode(text_result[0].tolist())))
 