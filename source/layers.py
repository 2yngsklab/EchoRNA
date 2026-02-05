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
from source.GVP import *
from source.util import *

class Normalize(nn.Module):
    """
    Layer normalization module with learnable gain and bias parameters.
    
    Args:
        features: Number of features to normalize
        epsilon: Small constant for numerical stability (default: 1e-6)
    """
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        """
        Apply normalization to input tensor.
        
        Args:
            x: Input tensor
            dim: Dimension to normalize over (default: -1)
            
        Returns:
            Normalized tensor
        """
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

class DihedralFeatures(nn.Module):
    """
    Embed dihedral angle features from protein/RNA backbone coordinates.
    
    Computes phi, psi, and omega dihedral angles and embeds them as
    sine/cosine features for use in structure-aware models.
    
    Args:
        node_embed_dim: Dimension of the output node embeddings
    """
    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X):
        """
        Featurize coordinates as dihedral angle embeddings.
        
        Args:
            X: Coordinates tensor of shape (batch_size, seq_len, 3, 3)
               representing backbone atoms
               
        Returns:
            Normalized dihedral angle embeddings
        """
        with torch.no_grad():
            V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        """
        Compute dihedral angles from backbone coordinates.
        
        Args:
            X: Backbone coordinates
            eps: Small epsilon for numerical stability
            return_angles: If True, return raw angles instead of sin/cos features
            
        Returns:
            Dihedral angle features (sin and cos of phi, psi, omega)
        """
        # First 3 coordinates are [N, CA, C] / [C4', C1', N1/N9]
        if len(X.shape) == 4:
            X = X[..., :3, :].reshape(X.shape[0], 3*X.shape[1], 3)
        else:
            X = X[:, :3, :]

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))

        # phi, psi, omega = torch.unbind(D,-1)
        #
        # if return_angles:
        #     return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

def geo_batch(batch):
    data_list = []
    i = 0
    data_list.append(torch_geometric.data.Data(
        coords=batch['coords'][i],  # num_res x 3 x 3
        node_s=batch['node_s'][i],  # num_res x num_conf x 4
        node_v=batch['node_v'][i],  # num_res x num_conf x 4 x 3
        edge_s=batch['edge_s'][i],  # num_edges x num_conf x 32
        edge_v=batch['edge_v'][i],  # num_edges x num_conf x 1 x 3
        edge_index=batch['edge_index'][i],  # 2 x num_edges
    ))
    return Batch.from_data_list(data_list)


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random Fourier features for encoding time steps.
    
    Maps scalar time values to high-dimensional features using random
    Fourier features with fixed (non-trainable) frequencies.
    
    Args:
        embed_dim: Dimension of the output embedding
        scale: Scale factor for random frequencies (default: 10.0)
    """  
    def __init__(self, embed_dim, scale=10.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        """
        Project time values to Fourier features.
        
        Args:
            x: Time values of shape (batch_size,)
            
        Returns:
            Fourier features of shape (batch_size, embed_dim)
        """
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TimeConditionedLayerNorm(nn.Module):
    """
    Layer normalization conditioned on time embeddings.
    
    Applies adaptive layer normalization where the scale and shift
    parameters are predicted from time embeddings.
    
    Args:
        hidden_size: Size of the input features
        time_embed_dim: Dimension of time embeddings
    """
    def __init__(self, hidden_size, time_embed_dim):
        super(TimeConditionedLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps = 1e-6,
                                        elementwise_affine = False) #Consider Changing to True
        self.fc_scale = nn.Linear(time_embed_dim, hidden_size)
        self.fc_shift = nn.Linear(time_embed_dim, hidden_size)
        self.silu = nn.SiLU()
        
        self.fc_scale.weight.data.fill_(0)
        self.fc_scale.bias.data.fill_(0)
        self.fc_shift.weight.data.fill_(0)
        self.fc_shift.bias.data.fill_(0)

    def forward(self, x, time_embed):
        """
        Apply time-conditioned layer normalization.
        
        Args:
            x: Input features of shape (batch_size, seq_len, hidden_size)
            time_embed: Time embeddings of shape (batch_size, time_embed_dim)
            
        Returns:
            Normalized and scaled features
        """
        normalized_x = self.layer_norm(x)
        scale = self.silu(self.fc_scale(time_embed)).unsqueeze(1)  # Adjust dimensions for broadcasting
        shift = self.silu(self.fc_shift(time_embed)).unsqueeze(1)  # Adjust dimensions for broadcasting

        return normalized_x * (1 + scale) + shift

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for attention mechanisms.
    
    Applies rotary positional embeddings by rotating query/key representations
    in a rotation matrix defined by position-dependent frequencies.
    
    Args:
        dim: Dimension of the embedding (should be even)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len):
        """
        Generate cosine and sine embeddings for given sequence length.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Tuple of (cos, sin) embeddings of shape (1, 1, seq_len, dim//2)
        """
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)  # [seq_len, dim//2]
        cos = freqs.cos()  # [seq_len, dim//2]
        sin = freqs.sin()  # [seq_len, dim//2]
        cos = cos.view(1, 1, seq_len, self.dim // 2)
        sin = sin.view(1, 1, seq_len, self.dim // 2)
        return cos, sin

class CrossAttention(nn.Module):
    """
    Cross-attention module with rotary positional embeddings.
    
    Implements cross-attention between RNA queries and protein keys/values
    with optional distance-based biasing.
    
    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        dropout_rate: Dropout probability (default: 0.1)
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key = nn.Linear(embed_dim, embed_dim, bias=True)
        self.value = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.out = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
    
    def apply_rotary_pos_emb(self, q, seq_len):
        """
        Apply rotary positional embeddings to query tensor.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Sequence length
            
        Returns:
            Query tensor with rotary embeddings applied
        """
        cos, sin = self.rotary_emb(seq_len)
        q_half_dim = self.head_dim // 2
        q1, q2 = q[..., :q_half_dim], q[..., q_half_dim:]
        q_embed = torch.cat([
            q1 * cos - q2 * sin,
            q2 * cos + q1 * sin
        ], dim=-1)
        
        return q_embed

    def forward(self, query, key, value, distance_bias=None, key_padding_mask=None):
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            key: Key tensor (batch_size, key_len, embed_dim)
            value: Value tensor (batch_size, key_len, embed_dim)
            distance_bias: Optional distance-based attention bias
            key_padding_mask: Optional mask for key padding
            
        Returns:
            output: Attention output (batch_size, query_len, embed_dim)
            attention_weights: Attention weights (batch_size, num_heads, query_len, key_len)
        """
        Q = self.query(query).view(query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(key.size(0), key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(value.size(0), value.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.apply_rotary_pos_emb(Q, query.size(1))
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if distance_bias is not None:
            gate = distance_bias.float() # Sigmoid gating
            scores = scores * gate  # Gate the attention scores

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(query.size(0), query.size(1), -1)
        output = self.out(output)

        return output, attention_weights

class TransformerBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio,
                 time_embed_dim,
                 dropout_rate=0.1,
                 **kwargs):
        super(TransformerBlock, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.layer_norm1 = TimeConditionedLayerNorm(hidden_size, time_embed_dim)
        # self.gvp_norm = TimeConditionedLayerNorm(hidden_size, time_embed_dim)
        self.gvp_projection = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attention1 = CrossAttention(hidden_size, num_heads, dropout_rate=dropout_rate)
        self.dropout_cross1 = nn.Dropout(dropout_rate)

        self.layer_norm2 = TimeConditionedLayerNorm(hidden_size, time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * mlp_ratio, hidden_size)
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def compute_distance_bias(self, distance_bias, batch_size, query_len, key_len, distance_bias_scale):
        if distance_bias is None:
            return None
        assert distance_bias.dim() == 3
        bias = distance_bias.unsqueeze(1).expand(-1, self.num_heads, query_len, key_len)  # Shape: [batch_size, num_heads, query_len, key_len]
        scale_factor = distance_bias_scale
        bias = bias * scale_factor
    
        return bias


    def forward(self, x, gvp_output, time_embed, mask_x=None, mask_gvp=None, mask_SA = None, distance_bias=None, distance_bias_scale=1):
        if mask_x is not None:
            mask_x_exp = mask_x.unsqueeze(-1).float() #expand to shape [batch_size, seq_len, 1]
        else:
            mask_x_exp = None
        x_cross_norm1 = self.layer_norm1(x, time_embed)  # Normalize RNA embeddings
        gvp_output_proj = self.gvp_projection(gvp_output)  # Project protein embeddings to transformer_dim
        gvp_output_proj = gvp_output_proj.unsqueeze(0).expand(x.size(0), -1, -1)  # Shape: [batch_size, protein_len, transformer_dim]
        # gvp_output_proj = self.gvp_norm(gvp_output_proj, time_embed)
        distance_bias = self.compute_distance_bias(
            distance_bias,  # [batch_size, rna_len, protein_len]
            batch_size=x.size(0),
            query_len=x.size(1),
            key_len=gvp_output_proj.size(1),
            distance_bias_scale=distance_bias_scale
        )
        x_cross1, weights = self.cross_attention1(
            query=x_cross_norm1,          # Shape: [batch_size, rna_len, transformer_dim]
            key=gvp_output_proj,          # Shape: [batch_size, protein_len, transformer_dim]
            value=gvp_output_proj,        # Shape: [batch_size, protein_len, transformer_dim]
            distance_bias=distance_bias,  # Shape: [batch_size, num_heads, rna_len, protein_len]
            key_padding_mask=mask_gvp         
        )
        x_cross1 = self.dropout_cross1(x_cross1)
        if mask_x_exp is not None:
            x = x * (1 - mask_x_exp) + (x + x_cross1) * mask_x_exp
        else:
            x = x + x_cross1
    
        x_norm2 = self.layer_norm2(x, time_embed) 
        x_mlp = self.mlp(x_norm2)
        x_mlp = self.dropout2(x_mlp)
        if mask_x_exp is not None:
            x = x * (1 - mask_x_exp) + (x + x_mlp) * mask_x_exp
        else:
            x = x + x_mlp
        return x, weights