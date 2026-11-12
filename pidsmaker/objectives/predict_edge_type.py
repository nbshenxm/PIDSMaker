import torch
import torch.nn as nn

from pidsmaker.utils.utils import compute_class_weights


class EdgeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, edge_type_dim, multi_edge, abstain=None):
        super(EdgeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.edge_type_dim = edge_type_dim
        self.multi_edge = multi_edge
        # CALIBER: abstention-trained decoder (gambler's loss, Liu et al., NeurIPS'19). The decoder
        # has one extra output, "abstain"; training minimizes -log(p_y + p_abstain / o), so the
        # model learns to abstain on events whose type it cannot predict better than 1/o instead
        # of spreading probability over them. abstain = {"o": payoff in (1, K], "warm": number of
        # plain cross-entropy training steps first}.
        self.abstain = abstain
        self._train_steps = 0

    def _abstain_forward(self, h, y, inference, class_weights):
        o = float(self.abstain.get("o", 2.0))
        logp = torch.log_softmax(h, dim=-1)                 # K real types + abstain
        lp_y = logp.gather(1, y[:, None]).squeeze(1)
        lp_m = logp[:, -1]
        log_keep = torch.log1p(-lp_m.exp().clamp(max=1 - 1e-6))
        if inference:
            # anomaly score = surprise among the real edge types, abstain mass removed
            return -(lp_y - log_keep), logp, lp_m, log_keep
        self._train_steps += 1
        if self._train_steps <= int(self.abstain.get("warm", 0)):
            per = -(lp_y - log_keep)
        else:
            per = -torch.logaddexp(lp_y, lp_m - torch.log(torch.tensor(o, device=h.device)))
        if class_weights is not None:
            w = class_weights[y]
            return (per * w).sum() / w.sum(), logp, lp_m, log_keep
        return per.mean(), logp, lp_m, log_keep

    def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
        h = self.decoder(h_src=h_src, h_dst=h_dst)

        class_weights = (
            compute_class_weights(edge_type, num_classes=self.edge_type_dim)
            if self.balanced_loss
            else None
        )

        if not self.multi_edge:
            edge_type = edge_type.argmax(dim=1)

        if self.abstain and not self.multi_edge:
            loss, logp, lp_m, log_keep = self._abstain_forward(h, edge_type, inference, class_weights)
            out = {"loss": loss}
            if inference:
                with torch.no_grad():
                    lq = logp[:, :-1] - log_keep[:, None]   # renormalized over the real types
                    out["entropy"] = -(lq.exp() * lq).sum(-1)
                    out["abstain"] = lp_m.exp()
            return out

        loss = self.loss_fn(h, edge_type, inference=inference, class_weights=class_weights)
        out = {"loss": loss}
        if inference:
            # CALIBER: entropy of the predicted edge-type distribution = the surprise the model
            # expects for this event (irreducible, aleatoric part); loss - entropy is the excess.
            with torch.no_grad():
                if self.multi_edge:
                    p = torch.sigmoid(h).clamp(1e-7, 1 - 1e-7)
                    ent = -(p * p.log() + (1 - p) * (1 - p).log()).sum(-1)
                else:
                    logp = torch.log_softmax(h, dim=-1)
                    ent = -(logp.exp() * logp).sum(-1)
            out["entropy"] = ent
        return out
