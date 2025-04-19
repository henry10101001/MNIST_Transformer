import argparse
import random
import torch
import torch.nn.functional as F
from mnist_transformer import MNISTTransformer, decode
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import matplotlib.pyplot as plt

# set device
# device = torch.device('cuda' if torch.cuda.is_available()
#                       else 'mps' if torch.backends.mps.is_available()
#                       else 'cpu')
device = torch.device('cpu')

print(f"Using device: {device}")

# load model checkpoint
def load_model(path):
    # load checkpoint and weights
    ckpt = torch.load(path, map_location=device)
    model = MNISTTransformer()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model

# main entrypoint
def main():
    # parse args
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/mnist_transformer_v0.6.pt')
    p.add_argument('--num',   type=int, default=10)
    args = p.parse_args()

    # load transformer
    model = load_model(args.model)
    print(next(model.parameters()).device)

    # interactive plotting
    plt.ion()
    rows = (args.num + 4) // 5
    cols = min(5, args.num)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    handles = []
    imgs = []

    # flatten axes list
    axes = axes.flatten() if args.num > 1 else [axes]

    # init blank images
    burnt_orange = [204, 85, 0]  # RGB for burnt orange
    for i, ax in enumerate(axes[:args.num]):
        img = [[[burnt_orange[0], burnt_orange[1], burnt_orange[2]] for _ in range(28)] for _ in range(28)]
        im = ax.imshow(img, cmap=None, vmin=0, vmax=255)
        ax.axis('off')
        handles.append(im)
        imgs.append(img)

    stop_requested = False
    # register quit key
    def on_key(event):
        nonlocal stop_requested
        if event.key == 'q':
            stop_requested = True
    fig.canvas.mpl_connect('key_press_event', on_key)

    # setup rich progress
    with Progress(
        TextColumn("[bold cyan]images {task.completed}/{task.total}"),
        BarColumn(),
        TextColumn(" • pixels {task.fields[p]}"),
        TimeElapsedColumn()
    ) as progress:
        img_task   = progress.add_task("gen imgs", total=args.num, p=0)
        pixel_task = progress.add_task("gen px",   total=784, p=0)

        # generate each image
        for i in range(args.num):
            # pick random class token
            cls = random.randint(0, 9)
            idx = torch.tensor([[cls]], device=device)

            # set the title to the target digit
            axes[i].set_title(f"Digit: {cls}", fontsize=14)

            # reset pixel progress
            progress.update(pixel_task, completed=0)

            # prepare blank array (burnt orange RGB)
            arr = [[[burnt_orange[0], burnt_orange[1], burnt_orange[2]] for _ in range(28)] for _ in range(28)]

            # stream tokens one by one
            for t in range(784):
                # forward only last seq_len
                cond = idx[:, -784:]
                # disable gradients for inference
                with torch.no_grad():
                    logits, _ = model(cond)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    nxt = torch.multinomial(probs, 1)
                idx = torch.cat((idx, nxt), dim=1)

                # write pixel
                val = decode(nxt)[0,0].item()
                r, c = divmod(t, 28)
                # set to black if val==0, else grayscale
                if val == 0:
                    arr[r][c] = [0, 0, 0]
                else:
                    arr[r][c] = [val, val, val]

                # update plot
                handles[i].set_data(arr)
                fig.canvas.draw_idle()
                plt.pause(0.001)

                # update pixel bar
                progress.update(pixel_task, advance=1, p=t+1)

            # finish this image
            progress.update(img_task, advance=1)
            # leave the image visible as we do the next one
            if stop_requested:
                break

    # disable interactive mode and show final grid
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()