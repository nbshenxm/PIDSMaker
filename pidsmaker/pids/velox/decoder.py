import torch.nn as nn

from pidsmaker.decoders.custom_edge_mlp_decoder import CustomEdgeMLP
from pidsmaker.decoders.custom_mlp_decoder import CustomMLPDecoder


class VeloxDecoder(nn.Module):
    """A lightweight decoder for Velox scoring."""

    def __init__(
        self,
        in_dim,
        out_dim,
        architecture="linear(0.5) | relu",
        dropout=0.0,
        edge_level=False,
        src_dst_projection_coef=1,
    ):
        super().__init__()
        self.edge_level = edge_level
        if edge_level:
            self.decoder = CustomEdgeMLP(
                in_dim=in_dim,
                out_dim=out_dim,
                architecture=architecture,
                dropout=dropout,
                src_dst_projection_coef=src_dst_projection_coef,
            )
        else:
            self.decoder = CustomMLPDecoder(
                in_dim=in_dim,
                out_dim=out_dim,
                architecture=architecture,
                dropout=dropout,
            )

    def forward(self, h=None, h_src=None, h_dst=None):
        if self.edge_level:
            return self.decoder(h_src=h_src, h_dst=h_dst)
        return self.decoder(h)
