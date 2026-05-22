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
from source.GVP import GVP
from source.util import label_smoothed_nll_loss
from source.lora import *
from source.layers import *

LIST_NODE_FEAT={"one-hot", "esm-2", "esm-if"}

class RNP_adapter(nn.Module):
    def __init__(self,
                 embed_dim=640, trans_dim=320, num_heads=20, num_trans_layers=8, mlp_ratio=4, time_dim=128, dropout_rate=0.05,
                 lora_config=None, lora_bias="lora_only", 
                 gvp_layers=4, node_feature=["one-hot", "esm-2", "esm-if"], device="cuda:0"):

        """
        args:
            embed_dim (int) : initial embedding of RNA-FM representation, default=640
            trans_dim (int) : hidden dimension of cross-attention layer
            num_head (int) : number of heads in cross-attention layer
            lora_config (dict) : configuration of LoRA
            mlp_ratio (int) : coefficient of linear hidden dimenstion multiplied insied cross-attention layer
            gvp_layers (int) : number of gvp-gnn layer
            node_feature (list) : type of node feature for protein, default = ["esm-2", "esm-if"]. 
                                  FYI, esm-2 has 1280, and esm-if has 512 dimenstion. Default is one-hot-encoding of amino acid, 
                                  which is 21 dimension, including unknown a.a.
        """
        
        super(RNP_adapter, self).__init__()

        self.lora_config = lora_config
        self.lora_bias = lora_bias
        self.rna_encoder, _ = fm.pretrained.rna_fm_t12()
        self.rna_encoder = self.rna_encoder.to(device)
        self.initialize_RNALM()
        # self.lm_head = self.rna_encoder.lm_head 
        self.lm_head = nn.Linear(embed_dim, 25)
        
        self.linear_input = nn.Sequential(
            nn.Linear(embed_dim, trans_dim)
        )

        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_dim),
                                   nn.Linear(time_dim, time_dim),
                                   nn.GELU(),
                                   nn.Linear(time_dim, time_dim),
                                   nn.GELU())
        
        self.Transformer = nn.ModuleList([
            TransformerBlock(trans_dim, num_heads, mlp_ratio, time_dim, dropout_rate=dropout_rate)
            for _ in range(num_trans_layers)])
        
        self.linear_output = nn.Sequential(
            nn.Linear(trans_dim, embed_dim)
        )
        
        ### GVP node configuration
        invalid_feats = set(node_feature) - LIST_NODE_FEAT
        if invalid_feats:
            raise ValueError(f"Invalid node feature(s): {invalid_feats}. Must be a subset of {LIST_NODE_FEAT}")

        node_in_feat = 0
        if "one-hot" in node_feature:
            node_in_feat += 21
        if "esm-2" in node_feature:
            node_in_feat += 1280
        if "esm-if" in node_feature:
            node_in_feat += 512

        ### GVP hyper-parameters
        self.node_in_dim = (node_in_feat, 4)  # node_in_dim
        self.node_h_dim = (256, 128)  # node_h_dim
        self.edge_in_dim = (48, 1)  # edge_in_dim
        self.edge_h_dim = (128, 1)  # edge_h_dim
        self.num_layers = gvp_layers #number of GVP-GNN layers in encoder/decoder
        self.dihedral_angle = True
        drop_rate = .1
        activations = (F.relu, F.relu)
        ### GVP layers
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=activations, vector_gate=True)
        )
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim,
                activations=activations, vector_gate=True)
        )
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(self.node_h_dim, self.edge_h_dim,
                         activations=activations, vector_gate=True,
                         drop_rate=drop_rate)
            for _ in range(self.num_layers))
        self.W_out = GVP(self.node_h_dim, (self.node_h_dim[0], 0), activations=activations)
        if self.dihedral_angle:
            self.embed_dihedral = DihedralFeatures(self.node_h_dim[0])
    
    def initialize_RNALM(self):
        """
        Initialize RNA language model with optional LoRA adaptation.
        
        If lora_config is None, freezes all RNA encoder parameters.
        Otherwise, applies LoRA and marks only LoRA parameters as trainable.
        """
        if self.lora_config == None:
            self.rna_encoder.eval()
            for param in self.rna_encoder.parameters():
                param.requires_grad = False
        else:
            self.rna_encoder = apply_lora_to_model(self.rna_encoder, self.lora_config)
            mark_only_lora_as_trainable(self.rna_encoder, bias=self.lora_bias)

    def struct_forward(self, batch, **kwargs):
        """
        Forward pass through protein structure encoder using GVP-GNN.
        
        Args:
            batch: PyTorch Geometric batch containing protein structure data
                   with node_s, node_v, edge_s, edge_v, edge_index, and coords
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            gvp_output: Protein structure embeddings from GVP encoder
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        if self.dihedral_angle:
            dihedral_feats = self.embed_dihedral(batch.coords).reshape_as(h_V[0])
            h_V = (h_V[0] + dihedral_feats, h_V[1])

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        gvp_output = self.W_out(h_V)
        return gvp_output

    def forward(self, protein, rna, t,
                num_timesteps=100,
                mask_x=None, 
                mask_gvp=None,
                distance_bias=None, distance_bias_scale=10,
                return_attn=False):
        """
        Forward pass through the RNA-protein adapter model.
        
        Args:
            protein: PyTorch Geometric batch with protein structure data
            rna: Dictionary containing 'input_ids' for RNA sequences
            t: Current diffusion timestep (int or tensor)
            num_timesteps: Total number of diffusion steps (default: 100)
            mask_x: Optional mask for RNA tokens
            mask_gvp: Optional mask for protein nodes
            distance_bias: Optional distance-based attention bias
            distance_bias_scale: Scaling factor for distance bias (default: 10)
            return_attn: Whether to return cross-attention weights (default: False)
            
        Returns:
            ans: Predicted logits for RNA tokens (batch_size, seq_len, vocab_size)
            fm_logits: RNA-FM language model logits
            cross_attention_weights: List of attention weights from each transformer layer
        """ 
        gvp_output = self.struct_forward(protein)
        x = self.rna_encoder(rna["input_ids"], repr_layers=[12])
        fm_logits, x = x['logits'], x["representations"][12]
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(rna["input_ids"].device)
        t = t.expand(x.shape[0])
        t = t / num_timesteps
        time_embed = self.time_embed(t)
        x = self.linear_input(x)   
        cross_attention_weights = [] 
        time_embed = time_embed*100
        for layer in self.Transformer:
            x, weights = layer(x, gvp_output, time_embed,
                               mask_x=mask_x,
                               mask_gvp=mask_gvp, 
                               distance_bias=distance_bias,
                               distance_bias_scale = distance_bias_scale)
            if return_attn:
                cross_attention_weights.append(weights)
        out = self.linear_output(x)
        ans = self.lm_head(out)
        return ans, fm_logits, cross_attention_weights