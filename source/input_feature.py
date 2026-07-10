#!/usr/bin/env python3

import pickle
from pathlib import Path

import esm
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from Bio.PDB.MMCIFParser import MMCIFParser

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

# =========================
# Geometry utilities
# =========================

def get_posenc(edge_index, num_posenc=16):
    """Sinusoidal positional encoding of each edge's sequence distance (src - dst)."""
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_posenc)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

def rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """Expand scalar distances D into Gaussian radial basis function features."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

def normalize(tensor, dim=-1):
    """Normalize a tensor to unit length along `dim`, mapping any NaNs to zero."""
    return torch.nan_to_num(
        torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True))
    )

def get_orientations_single(X):
    """Per-residue forward/backward unit vectors along the CA backbone (X: [N, 3])."""
    forward  = normalize(X[1:]  - X[:-1])
    backward = normalize(X[:-1] - X[1:])
    forward  = F.pad(forward,  [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], dim=-2)

def get_sidechains_single(X):
    """Per-residue sidechain orientation vectors from (N, CA, C) atoms (X: [N, 3, 3])."""
    p, origin, n = X[:, 0], X[:, 1], X[:, 2]
    n = normalize(n - origin)
    p = normalize(p - origin)
    return torch.cat([n.unsqueeze(-2), p.unsqueeze(-2)], dim=-2)

def construct_data_single(coords, seq, num_posenc=16, num_rbf=32, knn_num=10):
    """
    Build a KNN graph over the protein and compute its node/edge geometric features.

    Args:
        coords: [N, 3, 3] (N, CA, C) backbone coordinates.
        seq: protein sequence string (one-letter codes).
        num_posenc: edge positional-encoding dimension.
        num_rbf: edge distance RBF dimension.
        knn_num: number of nearest neighbors connected per node.

    Returns:
        dict: coords, node_s, node_v, edge_s, edge_v, edge_index for the graph.
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
    Extract one chain's sequence and (N, CA, C) backbone coordinates from a CIF file.

    Missing backbone atoms are filled with NaN. Raises ValueError if the chain has
    no parseable residues.

    Returns:
        dict: {"coords": Tensor[N, 3, 3], "seq": str}
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

    return {
        "coords": torch.tensor(np.array(protein_coord, dtype=np.float32)),
        "seq":    cif_protein_seq,
    }

def load_esm_models(device):
    """
    Load the pretrained ESM-IF1 (inverse folding) and ESM2 sequence models.

    Returns:
        tuple: (if_model, if_alphabet, esm2_model, esm2_batch_converter)
    """
    if_model, if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    if_model = if_model.to(device).eval()

    esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_model = esm2_model.to(device).eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()

    return if_model, if_alphabet, esm2_model, esm2_batch_converter

def generate_protein_embeddings(protein, protein_chain, if_model, if_alphabet,
                                esm2_model, esm2_batch_converter, device):
    """
    Compute ESM2 sequence and ESM-IF1 structure embeddings for one protein chain.

    Returns:
        tuple: (esm2_rep [1, L, D], if_rep [L, D']) tensors moved to CPU.
    """
    _, _, batch_tokens = esm2_batch_converter([("protein1", protein["seq"])])
    batch_tokens = batch_tokens.to(device)
    batch_tokens[batch_tokens == 24] = 3

    with torch.no_grad():
        results = esm2_model(batch_tokens, repr_layers=[33], return_contacts=True)
    esm2_rep = results["representations"][33][:, 1:-1, :]

    if_model = if_model.cpu()
    coords_dict = {protein_chain: protein["coords"]}
    if_rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(
        if_model, if_alphabet, coords_dict, protein_chain,
    )

    return esm2_rep.cpu(), if_rep.cpu()

# =========================
# Main preprocessing pipeline
# =========================

def preprocess_protein(protein_path, chain_id,
                       protein_pkl_path, esm2_pt_path, if_pt_path, device):
    """
    Parses a protein CIF file and generates ESM embeddings and a KNN graph.

    Args:
        protein_path: Path to the protein CIF file.
        chain_id: Chain ID to extract.
        protein_pkl_path: Output path for the KNN structural data (Pickle).
        esm2_pt_path: Output path for the ESM2 sequence embeddings.
        if_pt_path: Output path for the ESM-IF1 structure embeddings.
        device: torch device to run the embedding models on.

    Returns:
        tuple: (pkl_data dict, esm2_rep [L, D], if_rep [L, D'])
    """
    protein_path = Path(protein_path).resolve()
    for p in (protein_pkl_path, esm2_pt_path, if_pt_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing protein {protein_path.name}, chain {chain_id}")
    protein = parse_cif_protein(protein_path, chain_id)

    print("Loading ESM models and generating embeddings")
    if_model, if_alphabet, esm2_model, esm2_batch_converter = load_esm_models(device)
    esm2_rep, if_rep = generate_protein_embeddings(
        protein, chain_id,
        if_model, if_alphabet,
        esm2_model, esm2_batch_converter,
        device,
    )

    print("Constructing and saving protein data")
    pkl_data = construct_data_single(protein["coords"], protein["seq"], knn_num=12)
    pkl_data["attention_bias"] = None

    try:
        with open(protein_pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)
        torch.save(esm2_rep, esm2_pt_path)
        torch.save(if_rep,   if_pt_path)
        print(f"  Saved: {protein_pkl_path}")
        print(f"  Saved: {esm2_pt_path}")
        print(f"  Saved: {if_pt_path}")
    except OSError as e:
        print(f"  Warning: failed to save protein inputs ({e}); continuing with in-memory data")

    # Return the same object shapes as sample_EchoRNA.load_protein_inputs
    # (esm2 saved with a batch dim above, so squeezed here to [L, D])
    return pkl_data, esm2_rep.squeeze(), if_rep
