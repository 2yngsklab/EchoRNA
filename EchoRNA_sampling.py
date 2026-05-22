#!/usr/bin/env python3
 
import argparse
import pickle
import random
import sys
import yaml
import gc
from pathlib import Path
 
import esm
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from Bio.PDB.MMCIFParser import MMCIFParser
from source.lora import LoRA_Config
from source.diffusion import Echo_diffusion

import warnings
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
# =========================
# Constants
# =========================
 
PROTEIN_LETTER_TO_NUM = {
    'G': 0, 'A': 1, 'V': 2, 'I': 3, 'L': 4, 'F': 5, 'P': 6, 'M': 7, 'W': 8,
    'C': 9, 'S': 10, 'T': 11, 'N': 12, 'Q': 13, 'Y': 14, 'H': 15, 'D': 16,
    'E': 17, 'K': 18, 'R': 19, 'X': 20,
}
 
RES_NAMES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'UNK',
]
 
RES_NAMES_1 = 'ARNDCQEGHILKMFPSTWYVMX'
TO_1LETTER  = {aaa: a for a, aaa in zip(RES_NAMES_1, RES_NAMES)}
 
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
# Structure parsing
# =========================

def get_posenc(edge_index, num_posenc=16):
    """
    Generates sinusoidal positional encodings for edges based on sequence distance.
    
    Args:
        edge_index (Tensor): A 2xN tensor containing the indices of connected nodes.
        num_posenc (int): The number of positional encoding features to generate.
        
    Returns:
        Tensor: A tensor containing sine and cosine encodings for each edge.
    """
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_posenc)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

def rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """
    Computes Radial Basis Function (RBF) embeddings for scalar distances.
    
    Args:
        D (Tensor): Distances between nodes.
        D_min (float): Minimum distance for RBF centers.
        D_max (float): Maximum distance for RBF centers.
        D_count (int): Number of RBF centers (dimension of the embedding).
        
    Returns:
        Tensor: RBF embeddings of the given distances.
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

def normalize(tensor, dim=-1):
    """
    Safely normalizes a tensor along a specified dimension to unit length.
    
    Args:
        tensor (Tensor): The input tensor to normalize.
        dim (int): The dimension to compute the norm along.
        
    Returns:
        Tensor: The normalized tensor. Replaces NaNs with zeros.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True))
    )

def get_orientations_single(X):
    """
    Calculates forward and backward sequential directions along the protein backbone.
    
    Args:
        X (Tensor): N x 3 tensor of CA (Alpha-Carbon) coordinates.
        
    Returns:
        Tensor: Concatenated forward and backward normalized orientation vectors.
    """
    forward  = normalize(X[1:]  - X[:-1])
    backward = normalize(X[:-1] - X[1:])
    forward  = F.pad(forward,  [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], dim=-2)

def get_sidechains_single(X):
    """
    Calculates estimated sidechain orientation vectors based on N, CA, and C atoms.
    
    Args:
        X (Tensor): N x 3 x 3 tensor of (N, CA, C) coordinates for each residue.
        
    Returns:
        Tensor: Normalized direction vectors representing sidechain geometry.
    """
    p, origin, n = X[:, 0], X[:, 1], X[:, 2]
    n = normalize(n - origin)
    p = normalize(p - origin)
    return torch.cat([n.unsqueeze(-2), p.unsqueeze(-2)], dim=-2)

def construct_data_single(coords, seq, num_posenc=16, num_rbf=32, knn_num=10):
    """
    Constructs a K-Nearest Neighbors (KNN) graph representing the 3D protein structure.
    Calculates node and edge geometric features.
    
    Args:
        coords (array-like): N x 3 x 3 arrays of atom coordinates (N, CA, C).
        seq (str): The protein amino acid sequence as a string.
        num_posenc (int): Dimension of edge positional encoding.
        num_rbf (int): Dimension of edge distance RBF encoding.
        knn_num (int): Number of neighbors to connect for the spatial graph.
        
    Returns:
        dict: A dictionary containing node/edge features, indices, and coordinates
              ready to be formatted for PyTorch Geometric.
    """
    coords = torch.as_tensor(coords, dtype=torch.float32)
    seq = torch.as_tensor(
        [PROTEIN_LETTER_TO_NUM[residue] for residue in seq],
        dtype=torch.long,
    )
 
    coord_C    = coords[:, 1].clone()
    edge_index = torch_cluster.knn_graph(coord_C, k=knn_num)
    edge_index = torch_geometric.utils.coalesce(edge_index)
 
    orientations = get_orientations_single(coord_C)
    sidechains   = get_sidechains_single(coords)
 
    edge_vectors = coord_C[edge_index[0]] - coord_C[edge_index[1]]
    edge_rbf     = rbf(edge_vectors.norm(dim=-1), D_count=num_rbf)
    edge_posenc  = get_posenc(edge_index, num_posenc)
 
    node_s = (seq.unsqueeze(-1) == torch.arange(21).unsqueeze(0)).float()
    node_v = torch.cat([orientations, sidechains], dim=-2)
    edge_s = torch.cat([edge_rbf, edge_posenc], dim=-1)
    edge_v = normalize(edge_vectors).unsqueeze(-2)
 
    node_s, node_v, edge_s, edge_v = map(
        torch.nan_to_num,
        (node_s, node_v, edge_s, edge_v),
    )
 
    return {
        'coords':     coords,
        'node_s':     node_s,
        'node_v':     node_v,
        'edge_s':     edge_s,
        'edge_v':     edge_v,
        'edge_index': edge_index,
    }

# =========================
# Protein parsing and embedding
# =========================

def parse_cif_protein(cif_path, protein_chain):
    """
    Parses a mmCIF file to extract the 3D coordinates and sequence of a specific chain.
    
    Args:
        cif_path (str or Path): Path to the mmCIF structural file.
        protein_chain (str): The ID of the specific chain to extract.
        
    Returns:
        dict: A dictionary containing the sequence ("seq") and a tensor of 
              N, CA, C coordinates ("coords").
    """
    cif_parser = MMCIFParser(QUIET=True)
    structure = cif_parser.get_structure("cif", cif_path)[0]

    protein_coord = []
    cif_protein_seq = ""

    for chain in structure.get_chains():
        if chain.id != protein_chain:
            continue

        for res in chain.get_residues():
            if (res.id[0] == " " or res.id[0] == "H_MSE") and res.resname in TO_1LETTER:
                if "CA" in res:
                    CA_coord = res["CA"].get_coord()
                else:
                    CA_coord = np.array([float("nan"), float("nan"), float("nan")])

                if "N" in res:
                    N_coord = res["N"].get_coord()
                else:
                    N_coord = np.array([float("nan"), float("nan"), float("nan")])

                if "C" in res:
                    C_coord = res["C"].get_coord()
                else:
                    C_coord = np.array([float("nan"), float("nan"), float("nan")])

                res_coord = np.array([N_coord, CA_coord, C_coord])
                protein_coord.append(res_coord)
                cif_protein_seq += TO_1LETTER[res.get_resname()]

        break

    if len(protein_coord) == 0:
        raise ValueError(f"No residues found for chain '{protein_chain}' in {cif_path}")

    protein = {
        "distance_map": None,
        "coords": torch.tensor(np.array(protein_coord, dtype=np.float32)),
        "seq": cif_protein_seq
    }

    return protein
 
def load_esm_models(device):
    """
    Loads pretrained ESM language models (Inverse Folding and ESM2 sequence model).
    
    Args:
        device (torch.device): Device to load the models onto (CPU or GPU).
        
    Returns:
        tuple: (if_model, if_alphabet, esm2_model, esm2_batch_converter)
    """
    if_model, if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    if_model = if_model.to(device).eval()
 
    esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_model = esm2_model.to(device).eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()
 
    return if_model, if_alphabet, esm2_model, esm2_batch_converter
 
def generate_protein_embeddings(
    protein,
    protein_chain,
    if_model,
    if_alphabet,
    esm2_model,
    esm2_batch_converter,
    device
):
    """
    Generates representations for a given protein using both the ESM2 sequence model 
    and the ESM-IF1 structure model.
    
    Args:
        protein (dict): Extracted protein data containing 'seq' and 'coords'.
        protein_chain (str): Chain ID for the structure processing.
        if_model (nn.Module): Pretrained ESM-IF1 model.
        if_alphabet: Alphabet converter for ESM-IF1.
        esm2_model (nn.Module): Pretrained ESM2 sequence model.
        esm2_batch_converter: Batch converter for ESM2 sequences.
        device (torch.device): Execution device.
        
    Returns:
        tuple: (esm2_rep, if_rep) resulting representations stored on CPU.
    """
    _, _, batch_tokens = esm2_batch_converter([("protein1", protein["seq"])])
    batch_tokens = batch_tokens.to(device)
    batch_tokens[batch_tokens == 24] = 3
 
    with torch.no_grad():
        results = esm2_model(batch_tokens, repr_layers=[33], return_contacts=True)
    esm2_rep = results["representations"][33][:, 1:-1, :]
    
    if_model = if_model.cpu()
    coords_dict = {protein_chain: protein["coords"]}
    
    if_rep   = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(
        if_model,
        if_alphabet,
        coords_dict,
        protein_chain,
    )
 
    return esm2_rep.cpu(), if_rep.cpu()

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

def prepare_generation_inputs(protein_pkl_path, esm2_pt_path, if_pt_path,
                              rna_length, batch_size, device):
    """
    Loads previously saved geometry and embedding files, concatenates scalar features, 
    and initializes the blank RNA tensors for sequence generation.
    
    Args:
        protein_pkl_path (Path): Path to saved KNN structural data (Pickle).
        esm2_pt_path (Path): Path to saved ESM2 sequence embeddings.
        if_pt_path (Path): Path to saved ESM-IF1 structure embeddings.
        rna_length (int): The target length of the RNA sequence to be generated.
        device (torch.device): Device to place the tensors on.
        
    Returns:
        tuple: (protein_graph, rna_graph) formatted as PyTorch Geometric Data objects.
    """
    with open(protein_pkl_path, "rb") as f:
        protein = pickle.load(f)
 
    protein["node_s"] = torch.cat([
        protein["node_s"],
        torch.load(esm2_pt_path, weights_only=True).squeeze(),
        torch.load(if_pt_path, weights_only=True),
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
        batch_size (int): The number of independent sequences to generate.
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
    """
    Converts model token IDs back into string RNA sequences, trimming special 
    tokens (<cls>, <eos>).
    
    Args:
        toks_list (list): Lists of generated numeric tokens.
        
    Returns:
        list: Decoded string RNA sequences.
    """
    return ["".join(FM_TOK[t] for t in row[1:-1]) for row in tok_rows]

def write_fasta(sequences, output_fasta, name_prefix="seq"):
    """
    Writes a list of string sequences into a standard FASTA file format.
    
    Args:
        sequences (list): List of RNA string sequences.
        output_fasta (Path): Filepath to save the FASTA file.
        name_prefix (str): Prefix to prepend to each sequence's name in the file.
    """
    with open(output_fasta, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{name_prefix}{i}\n{seq}\n")


def main(protein_path, chain_id, output_dir, name,
         rna_length, num_sequence, sampling_strategy,
         random_seed, use_gpu,
         config, weight):
 
    # Validate inputs
    protein_path = Path(protein_path).resolve()
    if not protein_path.is_file():
        raise FileNotFoundError(f"Protein CIF file not found: {protein_path}")
 
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Setup Device
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
 
    # Prepare output subdirectories and file paths
    complex_dir = output_dir / "complex"
    esm2_dir    = output_dir / "esm2"
    if_dir      = output_dir / "esmIF"
    rna_dir     = output_dir / "RNA"
    for d in (complex_dir, esm2_dir, if_dir, rna_dir):
        d.mkdir(parents=True, exist_ok=True)
 
    sample_name      = name if name is not None else protein_path.stem
    protein_pkl_path = complex_dir / f"{sample_name}_{chain_id}.pickle"
    esm2_pt_path     = esm2_dir    / f"{sample_name}_{chain_id}.pt"
    if_pt_path       = if_dir      / f"{sample_name}_{chain_id}.pt"
    fasta_path       = rna_dir  / f"{sample_name}_{rna_length}_{random_seed}.fasta"
 
    #  Parse protein and generate embeddings 
    print(f"Parsing protein {protein_path.name}, chain {chain_id}")
    protein = parse_cif_protein(protein_path, chain_id)
 
    print(f"Loading ESM models and generating embeddings")
    if_model, if_alphabet, esm2_model, esm2_batch_converter = load_esm_models(device)
    esm2_rep, if_rep = generate_protein_embeddings(
        protein,
        chain_id,
        if_model,
        if_alphabet,
        esm2_model,
        esm2_batch_converter,
        device
    )
 
    # Build per-protein data bundle and persist intermediates    
    print("Constructing and saving protein data")
    pkl_data = construct_data_single(protein["coords"], protein["seq"], knn_num=12)
    pkl_data["attention_bias"] = None
 
    with open(protein_pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    torch.save(esm2_rep, esm2_pt_path)
    torch.save(if_rep,   if_pt_path)
    print(f"  Saved: {protein_pkl_path}")
    print(f"  Saved: {esm2_pt_path}")
    print(f"  Saved: {if_pt_path}")
 
    # Generation Step
    print(f"Loading diffusion model and generating {num_sequence} RNA(s) of length {rna_length}")
    net = load_generation_model(config, weight, device)
 
    protein_graph, rna_graph = prepare_generation_inputs(
        protein_pkl_path, esm2_pt_path, if_pt_path,
        rna_length, num_sequence, device,
    )
 
    tok_rows  = generate_rna_sequences(
        net, protein_graph, rna_graph, sampling_strategy=sampling_strategy,
    )
    
    sequences = tokens_to_sequences(tok_rows)[:num_sequence]
    write_fasta(sequences, fasta_path, name_prefix=f"{sample_name}_")
    
    print(f"Wrote {len(sequences)} RNA sequences to {fasta_path}")
    
    
    # clear cuda cache and memory
    if use_gpu and torch.cuda.is_available():

        try:
            del esm2_model
            del net
        except NameError:
            pass 

        gc.collect()
        torch.cuda.empty_cache()
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",  "--protein",           required=True,            help="Path to protein CIF file")
    parser.add_argument("-c",  "--chain",             required=True,           help="Protein chain ID")
    parser.add_argument("-d",  "--output-dir",        required=True,           help="Base output directory")
    parser.add_argument("-n",  "--name",              default=None,            help="Sample name (defaults to CIF stem)")
    parser.add_argument("-l",  "--rna-length",        default=20,   type=int,  help="Generated RNA length")
    parser.add_argument("-ns",  "--num-sequence",     default=100,  type=int,  help="Number of generated RNAs")
    parser.add_argument("-s",  "--sampling-strategy", default="vanilla",       help="Sampling strategy")
    parser.add_argument("-sd", "--random-seed",       default=42,   type=int,  help="Random seed")
    parser.add_argument("-g",  "--GPU",               default=None, nargs="?", const="cuda:0", help="Use GPU if available (default: CPU)")
    parser.add_argument("--config",                   default="./echorna_config.yaml",          help="Path to model configuration file")
    parser.add_argument("--weight",                   default="./echorna_weight.pth",           help="Path to the model weight")
    args = parser.parse_args()
 
    main(
        protein_path      = args.protein,
        chain_id          = args.chain,
        output_dir        = args.output_dir,
        name              = args.name,
        rna_length        = args.rna_length,
        num_sequence      = args.num_sequence,
        sampling_strategy = args.sampling_strategy,
        random_seed       = args.random_seed,
        use_gpu           = args.GPU,
        config            = args.config,            
        weight            = args.weight,
    )

