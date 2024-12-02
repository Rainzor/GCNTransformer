import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

import math
import numpy as np

from models.elements import PositionalEncoder, GELU, MLP
from models.gnn import GNN

class TransformerLayer(nn.Module):
    def __init__(self,
                d_model,
                n_head,
                dim_feedforward=2048,
                dropout=0.0,
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
                dropout=0.0):
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
        gnn_layers: int,
        transformer_width: int,
        transformer_layers: int,
        transformer_heads: int,
        hidden_dim: int,
        output_dim: int,
        embedding_dim: int,  # Number of frequency components for PositionalEncoder
        pos_dim: int,  # Dimension of positional features
        dropout: float = 0.1,
        gnn_type: str = 'GCNConv',
        pool: str = 'cls',
        patch_rw_dim=0
    ):
        super(GraphTransformer, self).__init__()
        assert pool in ['cls', 'mean'], "pool must be either 'cls' or 'mean'"
        self.num_features = num_features
        self.output_dim = output_dim
        self.pool = pool
        self.patch_rw_dim = patch_rw_dim
        self.gnn_layers = gnn_layers

        # Positional Encoder
        self.positional_encoder = PositionalEncoder(L=embedding_dim)
        self.proj_to_gnn = nn.Linear(num_features + embedding_dim*pos_dim, gnn_dim)
        # GCN module
        self.gnn = nn.ModuleList([
                    GNN(nin=gnn_dim, nout=gnn_dim, nlayer_gnn=1, gnn_type=gnn_type, bn=False, dropout=dropout)
                    for _ in range(gnn_layers)
                ])
        self.U = nn.ModuleList(
            [MLP(gnn_dim, gnn_dim, num_layers=1, with_final_activation=True, batch_norm=True) for _ in range(gnn_layers-1)])

        self.proj_to_transformer = nn.Linear(gnn_dim, transformer_width) if gnn_dim != transformer_width else nn.Identity()

        self.subgraph_pos_enc = nn.Linear(pos_dim * embedding_dim, transformer_width)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = nn.Linear(patch_rw_dim, transformer_width)

        # [CLS] token as a learnable embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_width))

        # If positional encoding increases the dimension, adjust the transformer width accordingly
        self.transformer = TransformerEncoder(
                            d_model=transformer_width,
                            n_head=transformer_heads,
                            num_layers=transformer_layers,
                            dim_feedforward=transformer_width*2
                            # dropout=dropout
                        )

      # Decoder with Batch Normalization
        self.decoder = MLP(transformer_width, hidden_dim, output_dim, num_layers=3, activation=GELU(), batch_norm=False, with_final_activation=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional Encoder weightare fixed (sinusoidal), no initialization needed
    
    def forward(self, data):
        """
        Forward pass through the model.

        Parameters:
        - data (Data): Graph data object containing node features, edge indices, positional features, etc.
        """
        B, P = data.mask.size()
        x = data.x
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch        
        pos = data.pos

        pos_enc = self.positional_encoder(pos)  # [total_num_nodes, embedding_dim * pos_dim]

        x = torch.cat([x, pos_enc], dim=-1)  # [total_num_nodes, num_features + embedding_dim * pos_dim]
        # Extract node features using GCN
        x = self.proj_to_gnn(x)  # [total_num_nodes, gnn_dim]
        x = x[data.subgraphs_nodes_mapper]  # [combined_num_nodes, gnn_dim]
        for i in range(self.gnn_layers):
            if i > 0:
                # scatter the node features to the subgraphs
                subgraph = scatter(x, batch_x, dim=0, reduce='mean',dim_size= P*B)[batch_x] # [combined_num_nodes, gnn_dim]
                # add subgraph global features to the node features
                x = x + self.U[i-1](subgraph)
                # scatter the node features back to the nodes
                # because some nodes may belong to multiple subgraphs
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
                
            x = self.gnn[i](x, edge_index)
        
        subgraph_x = scatter(x, batch_x, dim=0, reduce='mean',dim_size= B*P)  # [B*P, gnn_dim]

        # Project to Transformer input dimension
        subgraph_x = self.proj_to_transformer(subgraph_x)  # [B*P, transformer_width]
        sub_pos_enc = self.positional_encoder(data.patch_pos)  # [B*P, embedding_dim * pos_dim]
        subgraph_x += self.subgraph_pos_enc(sub_pos_enc)  # [B*P, transformer_width]
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(data.patch_pe)  # [B*P, transformer_width]

        # Reshape to patches
        trans_x = subgraph_x.view(B, P, -1)  # [batch_size, num_patches, transformer_width]
        trans_mask = data.mask  # [batch_size, num_patches]
        

        # Extract the graph's features
        graph_features = self._readout(trans_x, trans_mask, pool=self.pool)  # [batch_size, transformer_width]

        # Pass through the decoder
        out = self.decoder(graph_features)  # [batch_size, output_dim]

        return out  # [batch_size, output_dim]
    
    def _readout(self, x, mask, pool='cls'):
        # x: [batch_size, num_patchs, transformer_width]; 
        # mask: [batch_size, num_patchs, 1]
        B, P, _ = x.size()

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [batch_size, 1, transformer_width]

        x = torch.cat([cls_tokens, x], dim=1) if pool == 'cls' else x

        # Add True for [CLS] tokens
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, mask], dim=1) if pool == 'cls' else mask

        # Permute to match Transformer input shape (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # [1 + num_patchs, batch_size, transformer_width]

        # Initialize the attention mask with causal masking
        if pool == 'cls':
            attn_mask = torch.triu(torch.ones(P+1, P+1, device=x.device), diagonal=1).bool() 
            attn_mask[0, :] = False  # [CLS] can attend to all tokens including itself
        elif pool == 'mean':
            attn_mask = torch.triu(torch.ones(P, P, device=x.device), diagonal=1).bool()

        # Forward pass through Transformer
        x_transformed = self.transformer(x, 
                                        attn_mask= attn_mask,
                                        key_padding_mask=~key_padding_mask)  
                                        # [1 + num_patchs, batch_size, transformer_width]

        # Permute back to (batch_size, sequence_length, embedding_dim)
        x_transformed = x_transformed.permute(1, 0, 2)  # [batch_size, 1 + num_patchs, transformer_width]

        if pool == 'cls':
            # Extract the [CLS] token's features
            cls_features = x_transformed[:, 0, :]
        elif pool == 'mean':
            mask_unsqueeze = key_padding_mask.unsqueeze(-1).type_as(x_transformed)  # [batch_size, 1 + num_patchs, 1]
            sum_x = (x_dense * mask_unsqueeze).sum(dim=1)  # [batch_size, transformer_width]
            num_nodes = mask_unsqueeze.sum(dim=1)
            cls_features = sum_x / num_nodes.clamp(min=1)


        return cls_features  # [batch_size, transformer_width]

    def vmf_param(self, data):
        """
        Computes the von Mises-Fisher (vMF) parameters from the model output.
        """
        with torch.no_grad():
            # Forward pass through the model
            out = self(data)  # [batch_size, output_dim]
            
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
