import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

import math

from collections import OrderedDict

import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, L):
        """
        Initializes the PositionalEncoder module.

        Parameters:
        - L (int): Number of frequency components in the encoding.
        """
        super(PositionalEncoder, self).__init__()
        self.L = L

        # Precompute the frequency terms (2^k * pi)
        frequencies = torch.tensor([1 << i for i in range(L)], dtype=torch.float32) * math.pi
        self.register_buffer('frequencies', frequencies)  # Shape: (L,)

    def forward(self, p):
        """
        Computes the positional encoding gamma(p) for a given input p in the range [-1, 1].

        Parameters:
        - p (torch.Tensor): Input tensor of shape (batch_size, dim) with values in the range [-1, 1].

        Returns:
        - torch.Tensor: Positional encoding of shape (batch_size, dim * L).
        """
        # Ensure p is on the same device as frequencies
        # (Not needed since frequencies will be moved with the model)

        # Ensure p has the correct shape to broadcast with frequencies
        p = p.unsqueeze(-1)  # Shape: (batch_size, dim, 1)
        
        # Apply sin transformation only
        sin_encodings = torch.sin(p * self.frequencies)  # Shape: (batch_size, dim, L) 

        return sin_encodings.view(p.size(0), -1)  # Shape: (batch_size, dim * L)



class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

# GCN Module with Residual Connections
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        # GCN Layers
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.skip1 = nn.Linear(num_features, hidden_channels) if num_features != hidden_channels else nn.Identity()
        self.skip2 = nn.Identity()
        self.skip3 = nn.Identity()

    def forward(self, x, edge_index):
        # First Layer with Skip Connection
        residual = self.skip1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x + residual)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second Layer with Skip Connection
        residual = self.skip2(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x + residual)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Third Layer with Skip Connection
        residual = self.skip3(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x + residual)
        x = F.dropout(x, p=0.3, training=self.training)
        
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        if isinstance(self.skip1, nn.Linear):
            self.skip1.reset_parameters()




class GATBlock(nn.Module):
    def __init__(self, d_in, d_out, heads=8, dropout=0.3, activation=F.elu):
        super(GATBlock, self).__init__()
        
        assert d_out % heads == 0, "d_out must be divisible by heads"

        self.d_out = d_out
        self.num_heads = heads
        self.head_dim = d_out//heads


        # Multi-head GAT Layer
        self.attention = GATConv(
            d_in, 
            self.head_dim, 
            heads=self.num_heads, 
            dropout=dropout,
            residual=True
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)

        # Feedforward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_out, d_out*4),
            GELU(),
            nn.Linear(d_out*4, d_out)
        )       
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.norm1(x)
        x = self.attention(x,edge_index)
        x = self.norm2(x)
        x = x + self.dropout(self.feedforward(x))
        return x

class GNNEncoder(nn.Module):
    def __init__(self, num_features, emb_dim, num_layers=3, heads=8, dropout=0.3):
        super(GNNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        
        if num_features != emb_dim:
            self.Linear = nn.Linear(num_features, emb_dim)

        # 隐藏层
        for _ in range(num_layers):
            self.layers.append(GATBlock(
                d_in=emb_dim,
                d_out=emb_dim,
                heads=heads,
                dropout=dropout
            ))
        
    def forward(self, x, edge_index):
        if hasattr(self, 'Linear'):
            x = self.Linear(x)
            x = F.elu(x)

        for layer in self.layers:
            x = layer(x, edge_index)
        return x
    
    def reset_parameters(self):
        if hasattr(self, 'Linear'):
            self.Linear.reset_parameters()
        for layer in self.layers:
            layer.attention.reset_parameters()
            if isinstance(layer.feedforward, nn.Sequential):
                for module in layer.feedforward:
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()
            layer.norm1.reset_parameters()
            layer.norm2.reset_parameters()

class MLP(nn.Module):
    def __init__(self, num_features, hidden_channels, output_channels, num_layers=2, activation=F.relu):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_features, hidden_channels))
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, output_channels))
        self.activation = activation
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers-1:
                x = self.activation(x)
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

# Residual Attention Block
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4*d_model, d_model, num_layers=2, activation=GELU())
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        return self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# Transformer Module
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        """
        x: [sequence_length, batch_size, embedding_dim]
        key_padding_mask: [batch_size, sequence_length]
        """
        seq_length = x.size(0)
        device = x.device
        for block in self.resblocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x


class TransformerLayer(nn.Module):
    def __init__(self,
                d_model,
                n_head,
                dim_feedforward=2048,
                dropout=0.1,
                activation=F.relu):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head)

        self.feedforward = MLP(d_model, dim_feedforward, d_model, num_layers=2, activation=activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        return self.dropout1(x)

    def _ff_block(self, x):
        return self.dropout2(self.feedforward(x))

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x + self._sa_block(self.norm1(x), attn_mask, key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x
    
    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.feedforward.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

class TransformerEncoder(nn.Module):
    def __init__(self,
                d_model,
                n_head,
                num_layers,
                dim_feedforward=2048,
                dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_head, dim_feedforward, dropout, GELU())
            for _ in range(num_layers)
        ])
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x: [sequence_length, batch_size, embedding_dim]
        attn_mask: [sequence_length, sequence_length]
        key_padding_mask: [batch_size, sequence_length]
        """
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


# GraphTransformer class, mimicking Vision Transformer structure
class GraphTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        gnn_dim: int,
        transformer_width: int,
        transformer_layers: int,
        transformer_heads: int,
        hidden_dim: int,
        output_dim: int,
        embedding_dim: int,  # Number of frequency components for PositionalEncoder
        pos_dim: int,  # Dimension of positional features
        dropout: float = 0.1,
        pool: str = 'cls'
    ):
        super(GraphTransformer, self).__init__()
        assert pool in ['cls', 'mean'], "pool must be either 'cls' or 'mean'"
        self.num_features = num_features
        self.output_dim = output_dim
        self.pool = pool

        # Positional Encoder
        self.positional_encoder = PositionalEncoder(L=embedding_dim)
        self.proj_to_gnn = nn.Linear(num_features + embedding_dim*pos_dim, gnn_dim)
        # GCN module
        self.gnn = GNNEncoder(gnn_dim, gnn_dim)

        self.proj_to_transformer = nn.Linear(gnn_dim, transformer_width) if gnn_dim != transformer_width else nn.Identity()
        
        self.subgraph_pos_enc = nn.Linear(pos_dim * embedding_dim, transformer_width)

        # [CLS] token as a learnable embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_width))

        # # Mapping GCN output to Transformer input dimension
        # self.gcn_to_transformer = nn.Linear(embedding_dim*(pos_dim+1), transformer_width)

        # Layer normalization before Transformer
        self.ln_pre = nn.Identity()

        # If positional encoding increases the dimension, adjust the transformer width accordingly
        self.transformer = TransformerEncoder(
                            d_model=transformer_width,
                            n_head=transformer_heads,
                            num_layers=transformer_layers,
                            dim_feedforward=transformer_width*2,
                            dropout=dropout
                        )

        # Layer normalization after Transformer
        self.ln_post = nn.Identity()

      # Decoder with Batch Normalization
        self.decoder = MLP(transformer_width, hidden_dim, output_dim, num_layers=3, activation=GELU())

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional Encoder weightare fixed (sinusoidal), no initialization needed
    
    def forward(self, x, edge_index, p, batch=None):
        """
        x: [total_num_nodes, num_features]
        edge_index: [2, num_edges]
        p: [total_num_nodes, pos_dim], positional features for each node in the range [-1, 1]
        batch: [total_num_nodes], indicating the graph index each node belongs to
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        pos_enc = self.positional_encoder(p)  # [total_num_nodes, embedding_dim * pos_dim]

        x = torch.cat([x, pos_enc], dim=-1)  # [total_num_nodes, num_features + embedding_dim * pos_dim]
        # Extract node features using GCN
        x = self.proj_to_gnn(x)  # [total_num_nodes, gnn_dim]
        x = self.gnn(x, edge_index)  # [total_num_nodes, gnn_dim]
        x = self.proj_to_transformer(x)  # [total_num_nodes, transformer_width]

        x = x + self.subgraph_pos_enc(pos_enc)  # [total_num_nodes, transformer_width]

        # Extract the graph's features
        graph_features = self._read_out(x, batch, pool=self.pool)  # [batch_size, transformer_width]

        # Pass through the decoder
        out = self.decoder(graph_features)  # [batch_size, output_dim]

        return out  # [batch_size, output_dim]

    def _read_out(self, x, batch, pool='cls'):
        # x_dense: [batch_size, max_num_nodes, transformer_width]; 
        # mask: [batch_size, max_num_nodes]
        x_dense, mask = to_dense_batch(x, batch)  
        batch_size, max_num_nodes, _ = x_dense.size()

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, transformer_width]

        x_dense = torch.cat([cls_tokens, x_dense], dim=1)  # [batch_size, 1 + max_num_nodes, transformer_width]

        # Add True for [CLS] tokens
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, mask], dim=1)  # [batch_size, 1 + max_num_nodes]

        # Permute to match Transformer input shape (sequence_length, batch_size, embedding_dim)
        x_dense = x_dense.permute(1, 0, 2)  # [1 + max_num_nodes, batch_size, transformer_width]
        x_dense = self.ln_pre(x_dense)

        # Initialize the attention mask with causal masking
        attn_mask = torch.triu(torch.ones(max_num_nodes+1, max_num_nodes+1, device=x.device), diagonal=1).bool()

        # Allow [CLS] token (first token) to attend to all tokens
        attn_mask[0, :] = False  # [CLS] can attend to all tokens including itself
        # attn_mask[1:, 0] = True  # The other tokens cannot attend to [CLS]

        # attn_mask = None

        # Forward pass through Transformer
        x_transformed = self.transformer(x_dense, 
                                        attn_mask= attn_mask,
                                        key_padding_mask=~key_padding_mask)  
                                        # [1 + max_num_nodes, batch_size, transformer_width]

        # Permute back to (batch_size, sequence_length, embedding_dim)
        x_transformed = x_transformed.permute(1, 0, 2)  # [batch_size, 1 + max_num_nodes, transformer_width]

        if pool == 'cls':
            # Extract the [CLS] token's features
            cls_features = x_transformed[:, 0, :]
        elif pool == 'mean':
            mask_unsqueeze = key_padding_mask.unsqueeze(-1).type_as(x_transformed)  # [batch_size, 1 + max_num_nodes, 1]
            sum_x = (x_dense * mask_unsqueeze).sum(dim=1)  # [batch_size, transformer_width]
            num_nodes = mask_unsqueeze.sum(dim=1)
            cls_features = sum_x / num_nodes.clamp(min=1)
        else:
            raise ValueError('Invalid pooling type. Must be either "cls" or "mean".')

        # Extract the [CLS] token's features
        cls_features = self.ln_post(cls_features)  # [batch_size, transformer_width]

        return cls_features  # [batch_size, transformer_width]

    def vmf_param(self, x, edge_index, p, batch=None):
        """
        Computes the von Mises-Fisher (vMF) parameters from the model output.
        
        参数：
        - x (torch.Tensor): Feature tensor，Shape [total_num_nodes, num_features]
        - edge_index (torch.Tensor): Edge index tensor，Shape [2, num_edges]
        - p (torch.Tensor): Positional features tensor，Shape [total_num_nodes, pos_dim]
        - batch (torch.Tensor): Batch tensor，Shape [total_num_nodes]
        
        返回：
        - weights (torch.Tensor): [batch_size, num_vmf]
        - mus (torch.Tensor): [batch_size, num_vmf, 3]
        - kappas (torch.Tensor): [batch_size, num_vmf]
        """
        with torch.no_grad():
            # Forward pass through the model
            out = self(x, edge_index, p, batch)  # [batch_size, output_dim]
            
            # Batch size
            batch_size = out.size(0)  # [batch_size]
            output_dim = out.size(1)  # output_dim = num_vmf * 4
            assert output_dim % 4 == 0, "Output dimension must be divisible by 4 for vMF parameters"
            num_vmf = output_dim // 4  # vmf parameters: weights, kappas, theta, phi
            
            out = out.view(batch_size, num_vmf, 4)  # [batch_size, num_vmf, 4]
            
            # Get the vMF parameters
            weights = F.softmax(out[:, :, 0], dim=-1)  # [batch_size, num_vmf]
            kappas = torch.exp(out[:, :, 1])           # [batch_size, num_vmf]
            theta = torch.sigmoid(out[:, :, 2]) * math.pi  # [batch_size, num_vmf]
            phi = torch.sigmoid(out[:, :, 3]) * math.pi * 2  # [batch_size, num_vmf]
            
            # From spherical to Cartesian coordinates
            cos_theta = torch.cos(theta)  # [batch_size, num_vmf]
            sin_theta = torch.sin(theta)  # [batch_size, num_vmf]
            cos_phi = torch.cos(phi)      # [batch_size, num_vmf]
            sin_phi = torch.sin(phi)      # [batch_size, num_vmf]
            
            mus = torch.stack(
                (sin_theta * cos_phi, sin_theta * sin_phi, cos_theta),
                dim=-1
            )  # [batch_size, num_vmf, 3]
            
            return weights, mus, kappas
