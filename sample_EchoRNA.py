#!/usr/bin/env python3

import argparse
import pickle
import random
import yaml
import gc
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from source.lora import LoRA_Config
from source.diffusion import Echo_diffusion
from source.input_feature import preprocess_protein

import warnings
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

# =========================
# Constants
# =========================

FM_TOK = {
    0: '<cls>',
    1: '<pad>',
    2: '<eos>',
    3: '<unk>',
    4: 'A',
    5: 'C',
    6: 'G',
    7: 'U',
}

RNA_MASK_TOK = 24

# =========================
# Input file paths
# =========================

def get_input_paths(output_dir, sample_name, chain_id):
    """
    Single source of truth for the protein input file paths.

    These are always named "{sample_name}_{chain_id}", where sample_name is the
    input CIF file name without its extension (e.g. "1ivs_af3.cif" -> "1ivs_af3").

    Returns:
        tuple: (protein_pkl_path, esm2_pt_path, if_pt_path)
    """
    output_dir = Path(output_dir).resolve()
    protein_pkl_path = output_dir / "complex" / f"{sample_name}_{chain_id}.pickle"
    esm2_pt_path     = output_dir / "esm2"    / f"{sample_name}_{chain_id}.pt"
    if_pt_path       = output_dir / "esmIF"   / f"{sample_name}_{chain_id}.pt"
    return protein_pkl_path, esm2_pt_path, if_pt_path

def load_protein_inputs(protein_pkl_path, esm2_pt_path, if_pt_path):
    """
    Reads cached protein input files back into in-memory objects.

    The (pkl_data, esm2_rep, if_rep) tuple it returns has the exact same shapes
    as what preprocess_protein() returns, so the cache-hit and cache-miss paths
    feed prepare_generation_inputs() identical objects. The on-disk esm2 tensor
    is saved with a leading batch dim (see preprocess_protein), hence .squeeze().

    Returns:
        tuple: (pkl_data dict, esm2_rep [L, D], if_rep [L, D'])
    """
    with open(protein_pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    esm2_rep = torch.load(esm2_pt_path, weights_only=True).squeeze()
    if_rep   = torch.load(if_pt_path, weights_only=True)
    return pkl_data, esm2_rep, if_rep

# =========================
# Diffusion model
# =========================

def load_generation_model(config_path, checkpoint_path, device):
    """Load the diffusion model for inference.

    Args:
        config_path: Path to diffusion model YAML config
        checkpoint_path: Path to saved model checkpoint
        device: torch device to load model on

    Returns:
        net: The loaded diffusion model in eval mode
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lora_config = LoRA_Config(**config["lora_config"])
    adaptor_config = {
        **config["adaptor_config"],
        "lora_config": lora_config,
        "device":      device,
    }

    net = Echo_diffusion(adaptor_config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    return net

def prepare_generation_inputs(pkl_data, esm2_rep, if_rep,
                              rna_length, batch_size, device):
    """
    Concatenates the protein scalar features with the ESM embeddings and
    initializes the blank RNA tensors for sequence generation.

    Args:
        pkl_data (dict): KNN structural data (node/edge features, coords).
        esm2_rep (Tensor): ESM2 sequence embeddings, shape [L, D].
        if_rep (Tensor): ESM-IF1 structure embeddings, shape [L, D'].
        rna_length (int): The target length of the RNA sequence to be generated.
        batch_size (int): Number of independent sequences to generate.
        device (torch.device): Device to place the tensors on.

    Returns:
        tuple: (protein_graph, rna_graph) formatted as PyTorch Geometric Data objects.
    """
    protein = pkl_data

    protein["node_s"] = torch.cat([
        protein["node_s"],
        esm2_rep,
        if_rep,
    ], dim=1)
    protein["attention_bias"] = torch.zeros(rna_length, protein["coords"].size(0))

    seq_len = rna_length + 2  # CLS + sequence + EOS
    rna = {
        "input_ids": torch.full((batch_size, seq_len), RNA_MASK_TOK, dtype=torch.long, device=device),
        "pad_mask":  torch.full((batch_size, seq_len), True, dtype=torch.bool, device=device),
    }

    protein = torch_geometric.data.Data.from_dict(protein).to(device)
    rna     = torch_geometric.data.Data.from_dict(rna).to(device)
    return protein, rna

def generate_rna_sequences(net, protein, rna, sampling_strategy="vanilla"):
    """
    Executes the backward diffusion process to sample novel RNA sequences.

    Args:
        net (nn.Module): The loaded diffusion generation model.
        protein (Data): The featurized protein graph.
        rna (Data): The initial masked RNA template.
        sampling_strategy (str): Decoding sampling parameter (e.g., 'vanilla').

    Returns:
        list: A nested list of numeric token ids representing the generated RNAs.
    """
    tok, _ = net.generate_RDMSampling(
        (rna, protein),
        sampling_strategy=sampling_strategy,
        decoding_strategy="reparam-uncond-deterministic-cosine",
    )
    return tok.detach().cpu().numpy().tolist()

def tokens_to_sequences(tok_rows):
    """Convert model token IDs back into string RNA sequences, trimming <cls>/<eos>."""
    return ["".join(FM_TOK[t] for t in row[1:-1]) for row in tok_rows]

def write_fasta(sequences, output_fasta, name_prefix="seq"):
    """Write a list of string sequences into a standard FASTA file."""
    with open(output_fasta, "w", newline="\n") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{name_prefix}{i}\n{seq}\n")


def main(protein_path, chain_id, output_dir, name,
         num_sequence, rna_length, sampling_strategy,
         random_seed, use_gpu,
         config, weight):
    """
    Generate RNA sequences for a protein chain end to end: resolve and (re)use the
    cached protein inputs, run the diffusion model, and write the sampled sequences
    to a FASTA file under <output_dir>/RNA.
    """
    # Validate inputs
    protein_path = Path(protein_path).resolve()
    if not protein_path.is_file():
        raise FileNotFoundError(f"Protein CIF file not found: {protein_path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if use_gpu and torch.cuda.is_available():
        device = torch.device(use_gpu)
    elif use_gpu:
        device = torch.device("cpu")
        print("Warning: --GPU requested but CUDA not available; falling back to CPU.")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Seeding
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    # The sample name is the input CIF file name without its extension
    # "{sample_name}_{chain}", independent of the --name flag.
    sample_name = protein_path.stem
    protein_pkl_path, esm2_pt_path, if_pt_path = get_input_paths(output_dir, sample_name, chain_id)

    rna_dir = output_dir / "RNA"
    rna_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = rna_dir / (f"{name}.fasta" if name else f"{sample_name}.fasta")

    # Load cached protein inputs if present, otherwise generate (and cache) them.
    # Both branches yield the same (pkl_data, esm2_rep, if_rep) objects.
    if all(p.exists() for p in (protein_pkl_path, esm2_pt_path, if_pt_path)):
        print(f"Found cached protein inputs for {sample_name}_{chain_id}, skipping preprocessing")
        pkl_data, esm2_rep, if_rep = load_protein_inputs(protein_pkl_path, esm2_pt_path, if_pt_path)
    else:
        pkl_data, esm2_rep, if_rep = preprocess_protein(
            protein_path, chain_id, protein_pkl_path, esm2_pt_path, if_pt_path, device,
        )

    print(f"Loading diffusion model and generating {num_sequence} RNA sequences of length {rna_length}")
    net = load_generation_model(config, weight, device)

    protein_graph, rna_graph = prepare_generation_inputs(
        pkl_data, esm2_rep, if_rep,
        rna_length, num_sequence, device,
    )

    tok_rows  = generate_rna_sequences(
        net, protein_graph, rna_graph, sampling_strategy=sampling_strategy,
    )
    sequences = tokens_to_sequences(tok_rows)[:num_sequence]
    write_fasta(sequences, fasta_path, name_prefix=f"{sample_name}_")

    print(f"Wrote {len(sequences)} RNA sequences to {fasta_path}")

    # Clear CUDA cache and memory
    if use_gpu and torch.cuda.is_available():
        try:
            del net
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",  "--protein",            required=True,            help="Path to the structure file in CIF format")
    parser.add_argument("-c",  "--chain",              required=True,            help="Protein chain ID")
    parser.add_argument("-d",  "--output-dir",         default="./output",       help="Base output directory")
    parser.add_argument("-fn", "--name",               default=None,             help="Name of the output file (defaults to CIF stem)")
    parser.add_argument("-n",  "--num-sequence",       default=100,  type=int,   help="Number of generated RNAs")
    parser.add_argument("-l",  "--rna-length",         default=20,   type=int,   help="Generated RNA length")
    parser.add_argument("-s",  "--sampling-strategy",  default="vanilla", choices=["vanilla", "gumbel"], help="Sampling strategy: 'vanilla' (categorical sampling) or 'gumbel' (Gumbel noise)")
    parser.add_argument("-sd", "--random-seed",        default=42,   type=int,   help="Random seed")
    parser.add_argument("-g",  "--GPU",                default=None, nargs="?",  const="cuda:0", help="Use GPU if available (default: CPU)")
    parser.add_argument("--config",                    default="./echorna_config.yaml",  help="Path to model configuration file")
    parser.add_argument("--weight",                    default="./echorna_weight.pth",   help="Path to the model weight")
    args = parser.parse_args()

    main(
        protein_path      = args.protein,
        chain_id          = args.chain,
        output_dir        = args.output_dir,
        name              = args.name,
        num_sequence      = args.num_sequence,
        rna_length        = args.rna_length,
        sampling_strategy = args.sampling_strategy,
        random_seed       = args.random_seed,
        use_gpu           = args.GPU,
        config            = args.config,
        weight            = args.weight,
    )

