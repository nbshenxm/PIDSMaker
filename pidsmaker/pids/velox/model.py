"""Velox model implementation.

This module carries the Velox-specific model definition so the system can be
edited directly without going through the shared factory-based builder.
"""

import torch.nn as nn

from pidsmaker.model import Model
from .decoder import VeloxDecoder
from .encoder import VeloxEncoder
from .objectives import VeloxEdgeTypeObjective


class VeloxModel(Model):
    """A concrete Velox model built from explicit, editable components."""

    def __init__(self, cfg, in_dim, out_dim, device):
        edge_cfg = cfg.training.decoder.predict_edge_type
        encoder = VeloxEncoder(
            in_dim=in_dim,
            out_dim=cfg.training.node_out_dim,
            dropout=cfg.training.encoder.dropout,
        )
        decoder = VeloxDecoder(
            in_dim=cfg.training.node_out_dim,
            out_dim=cfg.dataset.num_edge_types,
            architecture=edge_cfg.edge_mlp.architecture_str,
            dropout=cfg.training.encoder.dropout,
            edge_level=True,
            src_dst_projection_coef=edge_cfg.edge_mlp.src_dst_projection_coef,
        )
        objective = VeloxEdgeTypeObjective(
            decoder=decoder,
            edge_type_dim=cfg.dataset.num_edge_types,
            balanced_loss=edge_cfg.balanced_loss,
        )
        super().__init__(
            encoder=encoder,
            objectives=[objective],
            objective_few_shot=None,
            device=device,
            is_running_mc_dropout=False,
            use_few_shot=False,
            freeze_encoder=False,
        )
