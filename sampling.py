#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_geometric
from torch_geometric.data import Batch
import numpy as np
import functools
import math
import fm
import os
import pandas as pd
import yaml
import argparse
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn
)
from rich.panel import Panel
from rich.text import Text

from source.dataloader import *
from source.lora import *
from source.diffusion import our_diffusion

# Initialize Rich console
console = Console()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensors_with_cls_eos_error(output):
    """Filter tensors that don't start with 0 or end with 2."""
    not_start_with_0 = output[:, 0] != 0
    not_end_with_2 = output[:, -1] != 2
    valid_rows = not_start_with_0 | not_end_with_2
    filtered_tensor = output[valid_rows]
    return filtered_tensor


def tensors_with_unusual_tokens(output):
    """Filter tensors with tokens not in the valid set."""
    valid_set = {0, 2, 4, 5, 6, 7}
    invalid_entries = ~torch.isin(output, torch.tensor(list(valid_set)))
    rows_with_invalid_entries = invalid_entries.any(dim=1)
    filtered_tensor = output[rows_with_invalid_entries]
    return filtered_tensor


def tensors_with_early_eos(output):
    """Filter tensors with early EOS tokens in the middle."""
    middle_columns = output[:, 1:-1]
    has_0_or_2_in_middle = (middle_columns == 0) | (middle_columns == 2)
    rows_with_0_or_2_in_middle = has_0_or_2_in_middle.any(dim=1)
    filtered_tensor = output[rows_with_0_or_2_in_middle]
    return filtered_tensor


def save_sequences_to_fasta(tensor, filename, alphabet):
    """Save tensor sequences to FASTA format."""
    with open(filename, 'w') as fasta_file:
        for i in range(tensor.size(0)):
            seq = tensor[i]
            output = ''
            for tok in seq:
                output += alphabet.all_toks[tok]
            trimmed_output = output[5:-5].replace('U', 'T')
            fasta_file.write(f">sequence_{i+1}\n{trimmed_output}\n")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_lora_config(config):
    """Create LoRA configuration from config dict."""
    lora_params = config['lora_config']
    return LoRA_Config(
        r=lora_params['r'],
        lora_alpha=lora_params['lora_alpha'],
        lora_dropout=lora_params['lora_dropout'],
        merge_weights=lora_params['merge_weights'],
        target_modules=lora_params['target_modules']
    )


def create_adaptor_config(config, lora_config, device):
    """Create adaptor configuration."""
    adaptor_params = config['adaptor_config']
    return {
        "embed_dim": adaptor_params['embed_dim'],
        "trans_dim": adaptor_params['trans_dim'],
        "num_heads": adaptor_params['num_heads'],
        "num_trans_layers": adaptor_params['num_trans_layers'],
        "mlp_ratio": adaptor_params['mlp_ratio'],
        "time_dim": adaptor_params['time_dim'],
        "dropout_rate": adaptor_params['dropout_rate'],
        "lora_config": lora_config,
        "lora_bias": adaptor_params['lora_bias'],
        "gvp_layers": adaptor_params['gvp_layers'],
        "node_feature": adaptor_params['node_feature'],
        "device": device
    }


def filter_sequences(sequences):
    """Apply all filtering steps to sequences."""
    # Filter unusual tokens
    mask = torch.ones(len(sequences), dtype=torch.bool)
    for row in tensors_with_unusual_tokens(sequences.cpu()):
        mask &= ~torch.all(sequences.cpu() == row, dim=1)
    out = sequences[mask].cpu()
    
    # Filter CLS/EOS errors
    mask = torch.ones(len(out), dtype=torch.bool)
    for row in tensors_with_cls_eos_error(out):
        mask &= ~torch.all(out == row, dim=1)
    out = out[mask]
    
    # Filter early EOS
    mask = torch.ones(len(out), dtype=torch.bool)
    for row in tensors_with_early_eos(out):
        mask &= ~torch.all(out == row, dim=1)
    out = out[mask]
    
    return out


def display_config_summary(config):
    """Display configuration summary in a nice table."""
    table = Table(title="Configuration Summary", show_header=True, header_style="bold blue")
    table.add_column("Parameter", style="dim", width=20)
    table.add_column("Value", style="bold")
    
    table.add_row("Device", str(config['device']))
    table.add_row("Seed", str(config['seed']))
    table.add_row("Dataset", config['dataset'])
    table.add_row("Batch Size", str(config['batch_size']))
    table.add_row("Epochs", str(config['epochs']))
    table.add_row("Lengths", str(config['lengths']))
    table.add_row("Model Tag", config['model_tag'])
    table.add_row("Sampling Strategy", config['sampling_strategy'])
    table.add_row("Decoding Strategy", config['decoding_strategy'])
    table.add_row("Save Directory", config['save_dir'])
    
    if config.get('complex_list'):
        table.add_row("Complex List", str(config['complex_list']))
    
    console.print(table)


def display_summary_report(generation_stats):
    """Display final summary report."""
    console.print("\n" + "="*60)
    console.print(Panel.fit("Generation Summary Report", style="bold green"))
    
    total_sequences = 0
    total_files = 0
    
    # Summary table
    summary_table = Table(title="Generation Statistics", show_header=True, header_style="bold cyan")
    summary_table.add_column("Epoch", style="dim")
    summary_table.add_column("Length", style="dim") 
    summary_table.add_column("Complex/Files", style="dim")
    summary_table.add_column("Total Sequences", justify="right", style="bold green")
    
    for epoch in sorted(generation_stats.keys()):
        for length in sorted(generation_stats[epoch].keys()):
            epoch_length_total = sum(generation_stats[epoch][length].values())
            file_count = len(generation_stats[epoch][length])
            
            summary_table.add_row(
                str(epoch),
                str(length),
                str(file_count),
                str(epoch_length_total)
            )
            
            total_sequences += epoch_length_total
            total_files += file_count
    
    console.print(summary_table)
    
    # Overall statistics
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="bold blue")
    stats_table.add_column("Value", style="bold green", justify="right")
    
    stats_table.add_row("Total Files Generated:", str(total_files))
    stats_table.add_row("Total Sequences Generated:", str(total_sequences))
    if total_files > 0:
        stats_table.add_row("Average Sequences per File:", f"{total_sequences/total_files:.1f}")
    
    console.print("\n")
    console.print(Panel(stats_table, title="Overall Statistics", style="bold blue"))
    
    # Detailed breakdown if requested
    if console.input("\n[bold yellow]Show detailed breakdown? (y/N): [/bold yellow]").lower() == 'y':
        detail_table = Table(title="Detailed Breakdown", show_header=True, header_style="bold magenta")
        detail_table.add_column("Epoch", style="dim")
        detail_table.add_column("Length", style="dim")
        detail_table.add_column("File", style="dim")
        detail_table.add_column("Sequences", justify="right", style="green")
        
        for epoch in sorted(generation_stats.keys()):
            for length in sorted(generation_stats[epoch].keys()):
                for filename, seq_count in generation_stats[epoch][length].items():
                    detail_table.add_row(str(epoch), str(length), filename, str(seq_count))
        
        console.print(detail_table)


def main():
    parser = argparse.ArgumentParser(description='EchoRNA Sampler')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Display welcome banner
    console.print(Panel.fit("EchoRNA Sampler", style="bold blue"))
    
    # Set up device and seed
    device = torch.device(config['device'])
    set_seed(config['seed'])
    
    # Display configuration
    display_config_summary(config)
    
    # Load alphabet
    console.print("[green]Loading RNA FM alphabet...[/green]")
    _, alphabet = fm.pretrained.rna_fm_t12()
    console.print("[green]Alphabet loaded successfully[/green]")
    
    # Create configurations
    lora_config = create_lora_config(config)
    adaptor_config = create_adaptor_config(config, lora_config, device)
    
    # Create save directory
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"[green]Save directory created: {save_dir}[/green]")
    
    # Load dataset
    dataset_name = config['dataset']
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    
    console.print(f"[green]Loading dataset: {dataset_name}...[/green]")
    test_dataset = RBPDataset(
        alphabet=alphabet,
        batch_size=batch_size,
        path=data_dir,
        dataset_dir=data_dir,
        split="test",
        device=device
    )
    console.print(f"[green]Dataset loaded: {len(test_dataset)} complexes[/green]")
    
    # Get configuration parameters
    epochs = config['epochs']
    lengths = config['lengths']
    complex_list = config.get('complex_list', [])
    model_tag = config['model_tag']
    sampling_strategy = config['sampling_strategy']
    decoding_strategy = config['decoding_strategy']
    checkpoint_base_path = config['checkpoint_base_path']
    
    # Statistics tracking
    generation_stats = defaultdict(lambda: defaultdict(dict))
    
    # Calculate total tasks for progress bar
    total_tasks = len(epochs) * len(lengths)
    if complex_list:
        total_tasks *= len(complex_list)
    else:
        total_tasks *= len(test_dataset)
    
    # Main processing with rich progress bars
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Main task tracker
        main_task = progress.add_task("[cyan]Overall Progress", total=total_tasks)
        
        for epoch in epochs:
            epoch_task = progress.add_task(f"[blue]Epoch {epoch}", total=len(lengths))
            
            for length in lengths:
                length_task = progress.add_task(f"[green]Length {length}", total=1)
                
                # Load model for current epoch
                checkpoint_path = f"{checkpoint_base_path}/model_epoch_{epoch}.pth"
                
                try:
                    progress.console.print(f"[yellow]Loading model epoch {epoch}...[/yellow]")
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    net = our_diffusion(adaptor_config).to(device)
                    net.load_state_dict(checkpoint)
                    progress.console.print(f"[green]Model epoch {epoch} loaded[/green]")
                    
                except FileNotFoundError:
                    progress.console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
                    progress.advance(epoch_task)
                    progress.advance(length_task)
                    # Skip all complexes for this epoch/length combination
                    if complex_list:
                        progress.advance(main_task, len(complex_list))
                    else:
                        progress.advance(main_task, len(test_dataset))
                    continue
                
                # Process complexes
                if complex_list:
                    complex_task = progress.add_task(f"[magenta]Processing complexes", total=len(complex_list))
                    
                    for complex_pdb in complex_list:
                        # Find matching complex in dataset
                        rbp = None
                        for complex_idx in range(len(test_dataset)):
                            if test_dataset[complex_idx][-1].upper() == complex_pdb[:4].upper():
                                rbp = test_dataset[complex_idx]
                                break
                        
                        if rbp is None:
                            progress.console.print(f"[yellow]Complex {complex_pdb} not found in dataset[/yellow]")
                            progress.advance(complex_task)
                            progress.advance(main_task)
                            continue
                        
                        # Generate filename
                        fasta_filename = f"{complex_pdb}_{model_tag}{epoch}_{length}.fasta"
                        fasta_path = os.path.join(save_dir, fasta_filename)
                        
                        # Modify input for specific lengths
                        if length != 'OG':
                            length_int = int(length)
                            rbp[0][0]['input_ids'] = torch.full((batch_size, length_int + 2), 24, 
                                                              dtype=torch.long, device=device)
                            rbp[0][0]['pad_mask'] = torch.full((batch_size, length_int + 2), True, 
                                                             dtype=torch.bool, device=device)
                        
                        # Generate sequences
                        progress.console.print(f"[cyan]Generating sequences for {complex_pdb}...[/cyan]")
                        res = net.generate_RDMSampling(
                            rbp[0],
                            sampling_strategy=sampling_strategy,
                            decoding_strategy=decoding_strategy
                        )
                        
                        # Filter sequences
                        filtered_sequences = filter_sequences(res[0])
                        
                        # Save to FASTA
                        if len(filtered_sequences) > 0:
                            save_sequences_to_fasta(filtered_sequences, fasta_path, alphabet)
                            generation_stats[epoch][length][fasta_filename] = len(filtered_sequences)
                            progress.console.print(f"[green]Saved {len(filtered_sequences)} sequences to {fasta_filename}[/green]")
                        else:
                            generation_stats[epoch][length][fasta_filename] = 0
                            progress.console.print(f"[yellow]No valid sequences for {fasta_filename}[/yellow]")
                        
                        progress.advance(complex_task)
                        progress.advance(main_task)
                    
                    progress.remove_task(complex_task)
                
                else:
                    # Process all complexes in dataset
                    dataset_task = progress.add_task(f"[magenta]Processing dataset", total=len(test_dataset))
                    
                    for complex_idx in range(len(test_dataset)):
                        rbp = test_dataset[complex_idx]
                        fasta_filename = f"{rbp[-1].upper()}_{model_tag}{epoch}_{length}.fasta"
                        fasta_path = os.path.join(save_dir, fasta_filename)
                        
                        # Modify input for specific lengths
                        if length != 'OG':
                            length_int = int(length)
                            rbp[0][0]['input_ids'] = torch.full((batch_size, length_int + 2), 24, 
                                                              dtype=torch.long, device=device)
                            rbp[0][0]['pad_mask'] = torch.full((batch_size, length_int + 2), True, 
                                                             dtype=torch.bool, device=device)
                        
                        # Generate sequences
                        progress.console.print(f"[cyan]Generating sequences for {rbp[-1].upper()}...[/cyan]")
                        res = net.generate_RDMSampling(
                            rbp[0],
                            sampling_strategy=sampling_strategy,
                            decoding_strategy=decoding_strategy
                        )
                        
                        # Filter sequences
                        filtered_sequences = filter_sequences(res[0])
                        
                        # Save to FASTA
                        if len(filtered_sequences) > 0:
                            save_sequences_to_fasta(filtered_sequences, fasta_path, alphabet)
                            generation_stats[epoch][length][fasta_filename] = len(filtered_sequences)
                            progress.console.print(f"[green]Saved {len(filtered_sequences)} sequences to {fasta_filename}[/green]")
                        else:
                            generation_stats[epoch][length][fasta_filename] = 0
                            progress.console.print(f"[yellow]No valid sequences for {fasta_filename}[/yellow]")
                        
                        progress.advance(dataset_task)
                        progress.advance(main_task)
                    
                    progress.remove_task(dataset_task)
                
                progress.advance(length_task)
                progress.remove_task(length_task)
            
            progress.advance(epoch_task)
            progress.remove_task(epoch_task)
    
    # Display final summary
    console.print("\n[bold green]Sampling completed successfully![/bold green]")
    display_summary_report(generation_stats)


if __name__ == "__main__":
    main()