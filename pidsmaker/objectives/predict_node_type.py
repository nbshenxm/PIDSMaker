import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.objectives.predict_edge_type import EdgeTypePrediction
from pidsmaker.utils.utils import compute_class_weights


class NodeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, node_type_dim, abstain=None, density=False):
        super(NodeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.node_type_dim = node_type_dim
        # CALIBER (default off; without the flags the forward pass is unchanged):
        # abstain = gambler's loss with one extra "abstain" output, as in EdgeTypePrediction; the
        # classifier output (the encoder output when decoder = none) then has node_type_dim + 1
        # units. density = class-conditional Gaussian with a shared covariance on the features
        # entering the classifier's last layer (hook attached by the factory through
        # attach_density_layer); novelty = -log density, novelty_y = half squared Mahalanobis
        # distance to the observed type's training mean.
        self.abstain = abstain
        self._train_steps = 0
        self.density = density
        self._collect = False
        self._dens = None
        self._acc = None
        self._z = None
        self.edge_type_dim = node_type_dim      # name used by the shared density helpers

    # density helpers shared with EdgeTypePrediction (they use only edge_type_dim/_acc/_dens/_collect)
    density_begin = EdgeTypePrediction.density_begin
    _accumulate = EdgeTypePrediction._accumulate
    density_end = EdgeTypePrediction.density_end
    _novelty = EdgeTypePrediction._novelty

    def attach_density_layer(self, layer):
        layer.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._z = inputs[0] if (self._collect or self._dens is not None) else None

    def _abstain_forward(self, h, y, inference, class_weights):
        o = float(self.abstain.get("o", 2.0))
        logp = torch.log_softmax(h, dim=-1)                 # K real types + abstain
        lp_y = logp.gather(1, y[:, None]).squeeze(1)
        lp_m = logp[:, -1]
        # surprise over the real types alone (abstain mass removed)
        lq_y = torch.log_softmax(h[:, :-1], dim=-1).gather(1, y[:, None]).squeeze(1)
        if inference:
            return -lq_y, lp_m
        self._train_steps += 1
        if self._train_steps <= int(self.abstain.get("warm", 0)):
            per = -lq_y
        else:
            per = -torch.logaddexp(lp_y, lp_m - float(np.log(o)))
            ce = float(self.abstain.get("ce", 0.0))
            if ce > 0:
                per = per + ce * (-lq_y)
        if class_weights is not None:
            w = class_weights[y]
            return (per * w).sum() / w.sum(), lp_m
        return per.mean(), lp_m

    def forward(self, h, node_type, inference, **kwargs):
        h_hat = self.decoder(h)

        class_weights = (
            compute_class_weights(node_type, num_classes=self.node_type_dim)
            if self.balanced_loss
            else None
        )

        node_type_classes = node_type.argmax(dim=1)

        if not (self.abstain or self.density):              # original behavior
            loss = self.loss_fn(h_hat, node_type_classes, inference=inference, weight=class_weights)
            return {"loss": loss, "out": F.log_softmax(h, dim=1)}

        nov = None
        if self.density and self._z is not None and self._z.shape[0] == h_hat.shape[0]:
            if self._collect:
                self._accumulate(self._z, node_type_classes)
            elif inference and self._dens is not None:
                nov = self._novelty(self._z, node_type_classes)
        self._z = None

        if self.abstain:
            loss, lp_m = self._abstain_forward(h_hat, node_type_classes, inference, class_weights)
            # ThreaTrace / Flash node scores read "out": keep it over the real types only
            out = {"loss": loss, "out": F.log_softmax(h[:, :-1], dim=1)}
            if inference:
                out["abstain"] = lp_m.exp().detach()
        else:
            loss = self.loss_fn(h_hat, node_type_classes, inference=inference, weight=class_weights)
            out = {"loss": loss, "out": F.log_softmax(h, dim=1)}
        if inference:
            with torch.no_grad():
                lq = torch.log_softmax(h_hat[:, :self.node_type_dim], dim=-1)
                out["entropy"] = -(lq.exp() * lq).sum(-1)
            if nov is not None:
                out["novelty"], out["novelty_y"] = nov
        return out
