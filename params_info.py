import torch
from rich.console import Console
from rich.table import Table
from collections import defaultdict
import argparse

from model import MNISTTransformer

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mnist_transformer_v1.5.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # set device and console
    device = torch.device(args.device)
    console = Console()
    console.print(f"Using device: [bold yellow]{device}[/bold yellow]")

    # load model
    console.print(f"Loading model [bold cyan]{args.model}[/]...")
    try:
        checkpoint = torch.load(f'models/{args.model}', map_location=device)
        console.print("[green]Checkpoint loaded successfully.[/green]")
        model = MNISTTransformer()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        console.print("[green]Model loaded and ready.[/green]")
    except FileNotFoundError:
        console.print(f"[bold red]Error: Model file not found at models/{args.model}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An error occurred while loading the model: {e}[/bold red]")

    # if loaded successfully
    if model:
        console.print("\n[bold underline]Model Parameter Statistics (Grouped by Module)[/bold underline]")
        table = Table(title="Module Parameter Information")
        table.add_column("Module Name", style="dim", width=40)
        table.add_column("Num Parameters", justify="right")
        table.add_column("Trainable Parameters", justify="right")
        table.add_column("% Trainable", justify="right")

        total_params = 0
        total_trainable_params = 0
        module_stats = defaultdict(lambda: {'total': 0, 'trainable': 0})

        for name, param in model.named_parameters():
            num_params = param.numel()
            is_trainable = param.requires_grad
            
            # extract top-level module name (e.g., 'embedding', 'layers.0', 'fc_out')
            module_name = name.split('.')[0]
            # if layers are nested (like transformer blocks), maybe group by the first two parts
            if module_name.startswith('layer') and '.' in name:
                parts = name.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    module_name = f"{parts[0]}.{parts[1]}" # e.g., layers.0
                else:
                    module_name = parts[0] # fallback to just 'layers'

            module_stats[module_name]['total'] += num_params
            if is_trainable:
                module_stats[module_name]['trainable'] += num_params

            total_params += num_params
            if is_trainable:
                total_trainable_params += num_params

        # sort modules by name for consistent output
        sorted_modules = sorted(module_stats.keys())

        for module_name in sorted_modules:
            stats = module_stats[module_name]
            mod_total = stats['total']
            mod_trainable = stats['trainable']
            percent_trainable = (mod_trainable / mod_total * 100) if mod_total > 0 else 0
            table.add_row(
                module_name,
                f"{mod_total:,}",
                f"{mod_trainable:,}",
                f"{percent_trainable:.2f}%"
            )

        console.print(table)
        console.print(f"\nTotal parameters: [bold blue]{total_params:,}[/bold blue]")
        console.print(f"Total trainable parameters: [bold green]{total_trainable_params:,}[/bold green]")
        non_trainable = total_params - total_trainable_params
        if non_trainable > 0:
            console.print(f"Non-trainable parameters: [bold red]{non_trainable:,}[/bold red]")

if __name__ == '__main__':
    main() 