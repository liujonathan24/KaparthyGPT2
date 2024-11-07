from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time



# input: input embedding

# GPT 2 is decoder only transformer (no encoder)
# No encoder side, no multi-head attention/cross-attention
# Most other is same.
# Difference 1: 
# Layer norms before and after last attention.
# Difference 2:

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

# CasualSelfAttention
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3 is ugly
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_heads = config.n_head
        self.n_embd = config.n_embd
        # not really a bias, more of a mask, but following the OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, 
                            config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size" (dimensionality per head), and C (number of channels)
        # = nh * hs
        # In GPT-2 (124M), n_heads = 12, hs=64, C = 12*64 = 768


        # math equivalent to normal multi-head in GPT 
        qkv = self.c_attn(x)
        q, k ,v  = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention, speeds up to 365 ms per batch (4), (hidden: 32), 

        """ # Swapping out for flash attention: stanford flash attention paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf
        # attention (matrerializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1) ))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        """


        # re-assemble all head outputs side by side/concatenation
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # output projection
        y = self.c_proj(y)
        return y
    



# MLP Block
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # similar to ReLU, but it's not exactly flat under 0.
        # issues known!! there's approximate and non-approximate (GPT2 is approximate)
        # Want GeLU because there's no change if something is negative (gradient = 0)
        # in paper: https://arxiv.org/pdf/1606.08415.pdf Section 3.1
        self.gelu = nn.GELU() 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Block

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) 
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Takes in input embeddings and applies two layer norms,
        one casual attention, and one MLP. The casual attention
        applies a mask to the attention weights so that the
        model can't see its own predictions, and the MLP applies
        two linear layers with a ReLU in between. The whole thing
        is residual, so the output is added to the input.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x





# Reflecting Hugging face transformer tensor sizes:

@dataclass
class GPT2Config:
    block_size: int = 1024 # Max token length
    vocab_size: int = 50257 #number of tokens, 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token 
    # ^ (very ugly number)
    n_layer: int = 12 # number of layers (suspicious)
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # nn.Embedding = fancy wrapper module around a single array of numbers
        # wraps a single tensor that allows you to access things by indexing rows
        self.transformer = nn.ModuleDict(dict(
            # word embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # positional encodings
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # h.0 to h.11 (h = hidden) module list
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # last layer norm
            ln_f = nn.LayerNorm(config.n_embd)

        ))
        # 758 back to vocab size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme from GPT-2 paper
        self.transformer.wte.weight = self.lm_head.weight
        # redirecting wte weight to the lm head :) copies data pointer.

        # init params
        self.apply(self._init_weights) # iterates all submodules and calls _init_weights() on each

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # twice because of 2: attention & MLP
                # scaled down by 1/sqrt(number of heads)

            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # std = 1/sqrt(number of stuff) (d_model = dimension model?)
            # d_model in language models are unsupervised multitask learners: gpt 2 paper: https://arxiv.org/pdf/2005.14165.pdf
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # not actually default, default is uniform
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # default initialization of layernorm is already correct :) 

    def forward(self, idx, target = None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # is broadcasting, each token gets identical position embedding of past words
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        # constructor/class method that returns the gpt object if we give it the model type.

        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s " % model_type)

        # n_layer, n_head and n_embd are determined from model type
        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params, 25 ugly >:(
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT-2 model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT-2 model checkpoints
        
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy over while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard this mask/buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # discard this mask/buffer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a conv1D module but we only wnat to use a vanilla linear layer
        # so we need to transpose these weights when we import them

        assert len(sd_keys) == len(sd_keys_hf), f"Models have {len(sd_keys)} and {len(sd_keys_hf)} parameters."
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the new transposed weights
                # they are saved with the format: "attn.c_attn.weight", we need to transpose them
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for key {k}: {sd_hf[k].shape[::-1]} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for key: {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        # at init, load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens)//(B*T)} batches") # epoch = 1 dataset / # per batch

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T) #input
        y = (buf[1:]).view(B, T) # output

        # advance the position
        self.current_position += B*T 
        # if loading the next batch is out of bounds, reset the position
        if self.current_position +(B*T+1)>= len(self.tokens):
            self.current_position = 0 # very simple reset
        return x, y






device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")
device = "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=32)

"""enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()

tokens = enc.encode(text[:1000])
B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
buf = buf.to(device) # must do this
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)"""

# get logits
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304)) # nicer number, should speed up gpu stuff
model.eval()
#model.to('cuda')
model = torch.compile(model) # massive speed boost supposedly!! for gpu :) takes longer to compile though. (a lot longer)
# ~450 to ~414 ish without gpu


# hyperparameters
max_lr = 6e-4
min_lr = max_lr * 0.01
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # linear warmup followed by cosine decay
    # 1) linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    #2) if it > lr)decay_iters, return min lr
    if it > max_steps:
        return min_lr
    #3) otherwise, use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0<=decay_ratio<=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


    # 2) cosine decay
    progress = (it - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + torch.cos(math.pi * progress))



dts = 0
# optimization time
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8)



for step in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad() # always zero-grad before each step

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # bfloat16 vs float16: https://pytorch.org/docs/stable/amp.html
        logits, loss = model(x,y)

    # without autocast: ~ 447.82  ms per step @ 4x batch size and 32x hidden size
    # with autocast: ~ 459 ms per step @ 4x batch size and 32x hidden size (probabily no change because no gpu lol)
    

    loss.backward() # adds to gradients, deposits += to past gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

    lr = get_lr(step) # get the cosine decay learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step() # step function (adamw) to update parameters and hopefully update

    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    dts += dt/50
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step: {step+1}, loss: {loss.item()}, lr: {lr:.4e}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
    # convert to a single float (loss.item)on CPU
    # loss itself is on GPU
print(dts)



import sys; sys.exit(0)





















# past tests

# Test 1
# model = GPT.from_pretrained('gpt2')
# print("Didn't crash yay!!")


# Test 2
num_return_sequences = 5
max_length = 30

# prefix tokens 
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hi, I'm a GPT-2 model.")
tokens = torch.tensor(tokens, dtype=torch.long) # 8
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # 5,8
x = tokens.to(device) #.to('cuda') by end will need to move to google colab :(

# generate
# right now, x is (Batch, Time) where B = 5, T = 8

# seed = 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, loss = model(x,y) # (B,T,vocab_size)
        #take the logits at the last position
        logits = logits[:, -1, :] # becomes (B,vocab_size) (only the last position!!!) inefficient, but correct
        # apply softmax to convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        #topk_probs becomes (5,50), topk_indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # top 50 tokens used, everything else probability clamped to 0
        # select token from top-k probabilities //sample from the distribution
        ix = torch.multinomial(topk_probs, num_samples=1) # 8,1
        # gather the corresponding indices/tokens
        xcol = torch.gather(topk_indices, -1, ix) # 8,1
        # append the new tokens to the sequence
        x = torch.cat([x, xcol], dim=1)

# print the generated text
for step in range(num_return_sequences):
    tokens = x[step, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)