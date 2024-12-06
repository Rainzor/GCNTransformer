import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
            Computes the positional encoding gamma(p) for a given input p.

            Parameters:
            - p (torch.Tensor): Input tensor of shape [..., dim]
            with values in the range [-1, 1] (though not strictly required).

            Returns:
            - torch.Tensor: Positional encoding of shape [..., dim * L].
            """
            # p: [..., dim]
            orig_shape = p.shape
            dim = orig_shape[-1]
            
            # Expand dimensions for broadcasting: [..., dim, 1]
            p = p.unsqueeze(-1)
            # frequencies: (L,)
            # After multiplication: [..., dim, L]

            sin_encodings = torch.sin(p * self.frequencies)  # [..., dim, L]

            # Reshape back to [..., dim * L]
            new_shape = orig_shape[:-1] + (dim * self.L,)
            return sin_encodings.view(new_shape)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

class MLP(nn.Module):
    def __init__(self, num_features, hidden_channels, output_channels = -1, num_layers=2, activation=F.relu, with_final_activation=False, normalization=None):
        super(MLP, self).__init__()
        if output_channels == -1:
            output_channels = hidden_channels
        self.num_layers = num_layers
        self.with_final_activation = with_final_activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_features, hidden_channels))
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, output_channels))

        self.norms = nn.ModuleList()
        if normalization is not None: 
            for i in range(num_layers-1):
                self.norms.append(normalization(hidden_channels))
            self.norms.append(normalization(output_channels))
        else:
            for i in range(num_layers):
                self.norms.append(nn.Identity())

        self.activation = activation

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers-1 or self.with_final_activation:
                x = self.norms[i](x)
                x = self.activation(x)
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        

