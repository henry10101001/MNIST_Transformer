import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import glob
import os
import argparse
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from model import MNISTTransformer, load_data, config

def train(device=torch.device('cpu')):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., cpu, cuda, mps)')
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    train_loader, val_loader = load_data()
    model = MNISTTransformer(device=device)
    model.to(device)
    
    # make dirs
    os.makedirs('stats', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

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
    total_samples = config['n_epochs'] * len(train_loader.dataset) # len(train_loader.dataset) is the number of samples in the training set
    task_id = progress.add_task(
        "training", total=total_samples, # use total samples for the progress bar total
        epoch=1, total_epochs=config['n_epochs'],
        train_loss=float('nan'), val_loss=float('nan'))

    with progress:
        for epoch in range(config['n_epochs']):
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
                
                if (step + 1) % config['bar_update_every'] == 0:
                    avg = window_loss / window_count
                    progress.update(task_id, train_loss=avg) # update the progress bar with the average loss for the window
                    window_loss, window_count = 0.0, 0
                    
                # advance progress bar by the number of samples in the batch
                progress.advance(task_id, advance=xb.size(0)) 
                
            # ---- Validation ----
            if (epoch + 1) % config['eval_every'] == 0:
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
                    f"[bold]epoch {epoch+1}/{config['n_epochs']}[/bold] "
                    f"tr avg: [blue]{epoch_tr_loss:.4f}[/blue] "
                    f"| val: [green]{epoch_val_loss:.4f}[/green]"
                )

            # ---- save checkpoint each epoch ----
            # get highest version number from existing checkpoints
            existing = glob.glob(f"models/mnist_transformer_v{config['main_version']}.*.pt")
            version = 0 if not existing else max([int(f.split(f"v{config['main_version']}.")[1].split(".pt")[0]) for f in existing]) + 1
            ckpt_name = f"models/mnist_transformer_v{config['main_version']}.{version}.pt"
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
    plt.title(f'mnist-transformer-v{config["main_version"]}.{version}')
    plt.savefig('stats/loss_plot.png'); plt.close()
    progress.print("loss plot saved to stats/loss_plot.png")
    
    return model

if __name__ == "__main__":
    train()