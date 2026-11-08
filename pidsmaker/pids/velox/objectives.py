import torch
import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.utils import compute_class_weights


class VeloxEdgeTypeObjective(nn.Module):
    """Velox edge-type prediction objective.

    This keeps Velox aligned with the original PIDSMaker configuration, where
    Velox inherits the edge-level `predict_edge_type` detector logic.
    """

    def __init__(self, decoder, edge_type_dim, balanced_loss=False):
        super().__init__()
        self.decoder = decoder
        self.edge_type_dim = edge_type_dim
        self.balanced_loss = balanced_loss

    def get_val_score(self):
        return 0.0

    def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
        logits = self.decoder(h_src=h_src, h_dst=h_dst)
        labels = edge_type.argmax(dim=1)
        weights = None
        if self.balanced_loss:
            weights = compute_class_weights(edge_type, num_classes=self.edge_type_dim)
            weights = weights.to(logits.device)

        per_edge_loss = F.cross_entropy(logits, labels, weight=weights, reduction="none")
        loss = per_edge_loss if inference else per_edge_loss.mean()

        return {
            "loss": loss,
            "out": F.log_softmax(logits, dim=1),
            "raw_logits": logits,
            "anomaly_score": per_edge_loss,
            "predicted_edge_type": logits.argmax(dim=1),
            "target_edge_type": labels,
        }
