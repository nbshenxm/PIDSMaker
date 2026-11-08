import torch.nn as nn

from pidsmaker.encoders.linear_encoder import LinearEncoder


class VeloxEncoder(nn.Module):
    """A simple, explicit encoder for Velox.

    The original Velox config in this repo uses a linear encoder with node features
    fed through a simple projection. This implementation exposes that logic as a
    concrete module so it can be modified directly.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.base_encoder = LinearEncoder(in_dim=in_dim, out_dim=out_dim, dropout=dropout)

    def forward(self, x, *args, **kwargs):
        return self.base_encoder(x, *args, **kwargs)
