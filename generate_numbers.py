import torch
import matplotlib.pyplot as plt
import numpy as np
from mnist_transformer import MNISTTransformer
import random

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model(model_path):
    print("Loading model...")
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    print("Checkpoint loaded")
    
    # Create a new model instance with the same hyperparameters
    model = MNISTTransformer()
    print("Model instance created")
    
    # Load the saved state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded and ready")
    return model

def decode(x):
    return x - 10

def generate_images(model, num_images=10):
    print(f"Generating {num_images} images...")
    # create a figure with subplots
    rows = (num_images + 4) // 5  # ceiling division to get number of rows needed
    cols = min(5, num_images)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if num_images > 5 else [axes] if num_images == 1 else axes
    
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}")
        # generate a random starting digit (0-9)
        cls = random.randint(0, 9)
        print(f"Selected class: {cls}")
        
        # Create input tensor with shape [1, 1] and add CLS token
        starter = torch.tensor([[cls]], device=device)
        print(f"Input tensor shape: {starter.shape}")
        
        # generate the tokens
        with torch.no_grad():
            print(f"Generating tokens for digit {cls}")
            try:
                generated = model.generate(starter)
                print(f"Generated tensor shape: {generated.shape}")
                # the output will be 784 tokens (28x28)
                generated_img = decode(generated[0]).cpu().numpy()
                print(f"Decoded image shape before reshape: {generated_img.shape}")
                if len(generated_img) > 784:  # ensure we only take the first 784 tokens
                    generated_img = generated_img[:784]
                generated_img = generated_img.reshape(28, 28)
                print(f"Final image shape: {generated_img.shape}")
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                raise
        
        # plot the image
        axes[i].imshow(generated_img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Generated from {cls}')
    
    plt.tight_layout()
    print("Displaying images...")
    plt.show()

def main():
    try:
        print("Starting main function")
        model = load_model('models/mnist_transformer_v0.1.pt')
        
        # generate 10 random number images
        generate_images(model, num_images=10)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main() 