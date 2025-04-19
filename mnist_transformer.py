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

# -- Hyperparameters --
main_version = 0
batch_size = 32
n_epochs = 10
seq_len = 784
n_heads = 6
n_layers = 6
n_embed = 300
ff_inner_multiplier = 4
dropout = 0.2
# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
vocab_size = 266 # 256 grayscale values + 10 for CLS tokens
learning_rate = 3e-4
weight_decay = 1e-2
bar_update_every = 50
eval_every = 1 # 1 epoch

device = torch.device('cpu')
print(f"Using device: {device}")


# -- Data Loading --

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

encode = lambda x: (x + 10) # add 10 to every grayscale value to account for CLS tokens 0-9; now, gs values are 10-265.
decode = lambda x: (x - 10)

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
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size, shuffle=False, drop_last=False)

# -- Feed Forward --
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, ff_inner_multiplier*n_embed),
            nn.ReLU(),
            nn.Linear(ff_inner_multiplier*n_embed, n_embed),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
            

# -- Attention --
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(dropout)
        
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
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
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
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x
        

# -- Transformer --
class MNISTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(seq_len, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.model_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embeddings = self.token_embedding_table(idx) # (batch_size, seq_len, n_embed) aka B, T, C where C = n_embed
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (seq_len, n_embed)
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
            idx_cond = idx[:, -seq_len:]
            logits, loss = self(idx_cond)
            # get just the last token's logits
            logits = logits[:, -1, :] # logits is now (B, C) where C = vocab_size
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx[:, 1:]  # remove the CLS token before returning
        
        
        
# -- Training Function --
def train():
    model = MNISTTransformer()
    model.to(device)
    
    # make dirs
    os.makedirs('stats', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # trackers
    train_losses, val_losses, epochs_done = [], [], []

    # rich progress bar
    progress = Progress(
        TextColumn("[bold cyan]epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.3f}%"),
        TextColumn("[bold]{task.completed:,} / {task.total:,} samples"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(" | tr\xa0avg: [blue]{task.fields[train_loss]:.4f}"),
        TextColumn(" | val: [green]{task.fields[val_loss]:.4f}"),
        transient=False
    )
    # calculate total samples instead of total steps
    total_samples = n_epochs * len(x_train) 
    task_id = progress.add_task(
        "training", total=total_samples, # use total samples for the progress bar total
        epoch=1, total_epochs=n_epochs,
        train_loss=float('nan'), val_loss=float('nan'))

    with progress:
        for epoch in range(n_epochs):
            # -- Training Loop --
            running_loss, running_items = 0.0, 0
            window_loss, window_count = 0.0, 0
            for step, seq in enumerate(train_loader):
                # seq is now the entire batch: shape (batch_size, seq_len+1)
                seq = seq.to(device) # take all 32 (or whatever) images

                xb, yb = seq[:, :-1], seq[:, 1:]
                _, loss = model(xb, yb)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * xb.size(0)
                running_items += xb.size(0)
                window_loss += loss.item() * xb.size(0)
                window_count += xb.size(0)
                
                if (step + 1) % bar_update_every == 0:
                    avg = window_loss / window_count
                    progress.update(task_id, train_loss=avg) # update the progress bar with the average loss for the window
                    window_loss, window_count = 0.0, 0
                    
                # advance progress bar by the number of samples in the batch
                progress.advance(task_id, advance=xb.size(0)) 
                
            # ---- Validation ----
            if (epoch + 1) % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_sum, val_items = 0.0, 0
                    for seq in val_loader:
                        seq = seq.to(device)
                        xb, yb = seq[:, :-1], seq[:, 1:]
                        _, vloss = model(xb, yb)
                        val_loss_sum += vloss.item() * xb.size(0)
                        val_items += xb.size(0)
                model.train()

                epoch_tr_loss = running_loss / running_items
                epoch_val_loss = val_loss_sum / val_items

                # stash for plots
                train_losses.append(epoch_tr_loss)
                val_losses.append(epoch_val_loss)
                epochs_done.append(epoch + 1)

                progress.update(
                    task_id,
                    epoch=epoch + 1,
                    train_loss=epoch_tr_loss, # update the progress bar with the average loss for the epoch, but this quickly gets replaced by the window loss on the next epoch
                    val_loss=epoch_val_loss
                )
                # use progress.print to avoid disrupting the bar
                progress.print(
                    f"[bold]epoch {epoch+1}/{n_epochs}[/bold] "
                    f"tr avg: [blue]{epoch_tr_loss:.4f}[/blue] "
                    f"| val: [green]{epoch_val_loss:.4f}[/green]"
                )

            # ---- save checkpoint each epoch ----
            # get highest version number from existing checkpoints
            existing = glob.glob("models/mnist_transformer_v0.*.pt")
            version = 0 if not existing else max([int(f.split("v0.")[1].split(".pt")[0]) for f in existing]) + 1
            ckpt_name = f"models/mnist_transformer_v{main_version}.{version}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1
            }, ckpt_name)
            progress.print(f"saved checkpoint to {ckpt_name}")
            
    # ---- plot losses ----
    plt.figure(figsize=(10,5))
    plt.plot(epochs_done, train_losses, label='train')
    plt.plot(epochs_done, val_losses,   label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.grid(); plt.legend()
    plt.title(f'mnist-transformer-v{main_version}.{version}')
    plt.savefig('stats/loss_plot.png'); plt.close()
    progress.print("loss plot saved to stats/loss_plot.png")
    
    return model

if __name__ == '__main__':
    train()