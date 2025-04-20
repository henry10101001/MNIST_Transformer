import glob
import keras
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from rich import print
from rich.progress import track
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import math
import time
import os
from torchvision import datasets, transforms
import argparse
from torch.nn.functional import scaled_dot_product_attention

# -- Hyperparameters --
config = {
    'main_version': 0,
    'batch_size': 32,
    'n_epochs': 10,
    'seq_len': 784,
    'n_heads': 6,
    'n_layers': 6,
    'n_embed': 300,
    'ff_inner_multiplier': 4,
    'dropout': 0.2,
    'vocab_size': 266, # 256 grayscale values + 10 for CLS tokens
    'learning_rate': 3e-4,
    'weight_decay': 1e-2,
    'bar_update_every': 50,
    'eval_every': 1 # 1 epoch
}

encode = lambda x: (x + 10) # add 10 to every grayscale value to account for CLS tokens 0-9; now, gs values are 10-265.
decode = lambda x: (x - 10)

def load_data():
    # download the dataset
    mnist_train = datasets.MNIST(
        root='data/',
        train=True,
        download=True,
    )
    mnist_test = datasets.MNIST(
        root='data/',
        train=False,
        download=True,
    )

    # process training data
    train_images = mnist_train.data.view(-1, 784).to(torch.long)          # (60000, 784), ensure long for embedding
    train_labels = mnist_train.targets.unsqueeze(1).to(torch.long)        # (60000, 1)
    train_images_encoded = encode(train_images)                           # encoding (offset 10 to every grayscale value)
    x_train_full = torch.cat([train_labels, train_images_encoded], dim=1) # (60000, 785) - Renamed temporarily

    # process test data
    test_images = mnist_test.data.view(-1, 784).to(torch.long)    # (10000, 784), ensure long for embedding
    test_labels = mnist_test.targets.unsqueeze(1).to(torch.long)  # (10000, 1)
    test_images_encoded = encode(test_images)                     # encoding
    x_test = torch.cat([test_labels, test_images_encoded], dim=1) # (10000, 785)


    # train/val split (50000, 785) (10000, 785)
    train_subset, val_subset = torch.utils.data.random_split(x_train_full, [50000, 10000])
    x_train = x_train_full[train_subset.indices]
    x_val = x_train_full[val_subset.indices]

    # -- Data Loader / Batching --
    train_loader = torch.utils.data.DataLoader(x_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    
    return train_loader, val_loader

# -- Feed Forward --
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, config['ff_inner_multiplier']*n_embed),
            nn.ReLU(),
            nn.Linear(config['ff_inner_multiplier']*n_embed, n_embed),
            nn.Dropout(config['dropout']),
        )
        
    def forward(self, x):
        return self.net(x)
            

# -- Attention --
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config['n_embed'], config['head_size'], bias=False)
        self.query = nn.Linear(config['n_embed'], config['head_size'], bias=False)
        self.value = nn.Linear(config['n_embed'], config['head_size'], bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['seq_len'], config['seq_len'])))
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        weights = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T) explanation: basically, for each batch, we multiply a (T, head_size) matrix by a (head_size, T) matrix, which results in a (T, T) matrix for each batch, so we get a (B, T, T) matrix
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # piece de resistance of ye olde 'decoder' transformer (AR)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        
        v = self.value(x) # (B, T, head_size)
        out = weights @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config['n_embed'], config['n_embed'])
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
# -- Block --
class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(config['n_embed'])
        self.ln1 = nn.LayerNorm(config['n_embed'])
        self.ln2 = nn.LayerNorm(config['n_embed'])
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x
        

# -- Transformer --
class MNISTTransformer(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embed'])
        self.position_embedding_table = nn.Embedding(config['seq_len'], config['n_embed'])
        self.blocks = nn.Sequential(*[Block(config['n_embed'], config['n_heads']) for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['n_embed'])
        self.model_head = nn.Linear(config['n_embed'], config['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embeddings = self.token_embedding_table(idx) # (batch_size, seq_len, n_embed) aka B, T, C where C = n_embed
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device)) # use self.device here
        x = token_embeddings + position_embeddings # (batch_size, seq_len, n_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.model_head(x)                # (batch_size, seq_len, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C) # Use reshape instead of view
            targets = targets.reshape(B*T)  # Use reshape instead of view
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens=784):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            print(f"Generating token {_ + 1} of {max_new_tokens}")
            idx_cond = idx[:, -config['seq_len']:]
            logits, loss = self(idx_cond)
            # get just the last token's logits
            logits = logits[:, -1, :] # logits is now (B, C) where C = vocab_size
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx[:, 1:]  # remove the CLS token before returning