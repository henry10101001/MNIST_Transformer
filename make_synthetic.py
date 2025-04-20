import argparse
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from model import MNISTTransformer, decode
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
import matplotlib.pyplot as plt
import os
import time

# load model checkpoint
def load_model(path, device):
    # load checkpoint and weights
    ckpt = torch.load(path, map_location=device)
    model = MNISTTransformer(device=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model

def main():
    # use known model as default
    model_path = f'mnist_transformer_3.5M_v1.1.pt'
    model_folder = 'models'

    # parse args
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=model_path)
    p.add_argument('--num',   type=int, default=10)
    p.add_argument('--device', default='cpu', help='Device to use (e.g., cpu, cuda, mps)')
    p.add_argument('--display', default=False, help='Dynamically display the generated images')
    args = p.parse_args()

    # set device based on args
    device = torch.device(args.device)
    print(f"Using inference device: {device}")

    # load transformer
    model = load_model(f'{model_folder}/{args.model}', device)
    print(next(model.parameters()).device)

    # interactive plotting
    if args.display:
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
    if args.display:
        for i, ax in enumerate(axes[:args.num]):
            img = [[[burnt_orange[0], burnt_orange[1], burnt_orange[2]] for _ in range(28)] for _ in range(28)]
            im = ax.imshow(img, cmap=None, vmin=0, vmax=255)
            ax.axis('off')
            handles.append(im)
            imgs.append(img)

    # create dir if needed
    os.makedirs("generated_images", exist_ok=True)
    
    # one single Progress for both Images and Pixels
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("• {task.fields[speed]} •"),
        TimeElapsedColumn(),
    )

    with progress:
        start_time = time.time()
        total_pixels = 0
        img_task = progress.add_task("Images", total=args.num, speed="0.00 s/img")
        pix_task = progress.add_task("Pixels", total=784, speed="0 px/s")
        
        # generate each image
        for img_num in range(args.num):
            # pick random class token
            cls = random.randint(0, 9)
            idx = torch.tensor([[cls]], device=device)

            # set the title to the target digit
            if args.display:
                axes[img_num].set_title(f"Digit: {cls}", fontsize=14)

            # create image buffer
            arr = np.full((28, 28, 3), burnt_orange, dtype=np.uint8)

            # stream tokens one by one
            for t in range(784):
                # forward only last seq_len (shouldn't be needed since we have fixed image sizes)
                cond = idx[:, -784:]
                # disable gradients for inference
                with torch.no_grad():
                    logits, _ = model(cond)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    nxt = torch.multinomial(probs, 1)
                idx = torch.cat((idx, nxt), dim=1)

                # decode and set pixel
                val = decode(nxt)[0, 0].item()
                r, c = divmod(t, 28)
                arr[r, c] = [val, val, val]

                if args.display:
                    handles[img_num].set_data(arr)
                    fig.canvas.draw_idle()
                    plt.pause(0.001)  # force render

                # update pixel progress
                total_pixels += 1  # count every pixel
                elapsed_total = time.time() - start_time
                avg_pix_speed = total_pixels / elapsed_total if elapsed_total > 0 else 0
                if t % 2 == 0:  # update display less frequently for performance
                    progress.update(
                        pix_task,
                        completed=t + 1,
                        speed=f"{avg_pix_speed:.1f} px/s"
                    )
            
            # update image progress
            elapsed_img = time.time() - start_time
            avg_img_time = elapsed_img / (img_num + 1) # +1 for the current image

            progress.update(
                img_task,
                advance=1,
                speed=f"{avg_img_time:.2f} s/img"
            )

            # reset pixel progress count but keep total_pixels for speed calculation
            progress.reset(pix_task, completed=0, total=784)

            # save image
            # find next available number for this class
            num = 0
            while os.path.exists(f'generated_images/image_{cls}_{num}.png'): # optimize this lol
                num += 1
            plt.imsave(f'generated_images/image_{cls}_{num}.png', arr.astype(np.uint8))

    # disable interactive mode and show final grid
    if args.display:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()