import numpy as np
import torch
import torch.nn as nn

from pidsmaker.utils.utils import compute_class_weights


class EdgeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, edge_type_dim, multi_edge, abstain=None,
                 density=False):
        super(EdgeTypePrediction, self).__init__()
        self.decoder = decoder
        # CALIBER: novelty = how unlike the TRAINING events an event's representation is.
        # A frozen pass over the training graphs before each evaluation fits a Gaussian per edge
        # type with a shared covariance on the decoder's hidden features (the input of its output
        # layer); at inference, novelty = -log density under that mixture (Lee et al., NeurIPS'18;
        # Mukhoti et al., CVPR'23). The abstain output says what is inherently unpredictable in
        # the training data (routine noise); novelty says what lies outside it.
        self.density = density
        self._collect = False
        self._dens = None
        self._acc = None
        self._z = None
        if density:
            self._out_layer().register_forward_hook(self._grab)
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.edge_type_dim = edge_type_dim
        self.multi_edge = multi_edge
        # CALIBER: abstention-trained decoder (gambler's loss, Liu et al., NeurIPS'19). The decoder
        # has one extra output, "abstain"; training minimizes -log(p_y + p_abstain / o), so the
        # model learns to abstain on events whose type it cannot predict better than 1/o instead
        # of spreading probability over them. abstain = {"o": payoff in (1, K], "warm": number of
        # plain cross-entropy training steps first}.
        # "warm": "auto" -- plain cross-entropy until the running training loss falls below log o,
        # the loss of abstaining on every event. Before that point abstaining everywhere is the
        # gambler's loss optimum, and a slowly learning model can lock into it (Liu et al. warm up
        # with cross-entropy for the same reason); the switch is permanent.
        self.abstain = abstain
        self._train_steps = 0
        self._warm_ema = None
        self._engaged = not (abstain and abstain.get("warm") == "auto")
        # "cov": target coverage (share of training events the model predicts rather than
        # abstains on). The payoff o is then adapted by integral control during training,
        # log o <- log o + k (abstain rate - (1 - cov)), clipped to (1, K], so that every dataset
        # abstains on the same share of events instead of whatever a fixed o produces.
        if abstain and "cov" in abstain:
            o0 = float(abstain.get("o", max(2.0, edge_type_dim / 2.0)))
            self.register_buffer("log_o", torch.tensor(float(np.log(o0))))
        else:
            self.log_o = None

    # ------------------------------------------------------------------ CALIBER: novelty density
    def _out_layer(self):
        d = self.decoder
        seq = getattr(getattr(d, "mlp", None), "mlp", None)
        if isinstance(seq, nn.Sequential) and isinstance(seq[-1], nn.Linear):
            return seq[-1]
        if isinstance(getattr(d, "lin_final", None), nn.Linear):
            return d.lin_final
        raise ValueError(f"--density: no output layer found in {type(d).__name__}")

    def _grab(self, module, inputs, output):
        self._z = inputs[0] if (self._collect or self._dens is not None) else None

    def density_begin(self):
        self._collect, self._dens, self._acc = True, None, None

    @torch.no_grad()
    def _accumulate(self, z, y):
        z = z.detach().double()
        K, D = self.edge_type_dim, z.shape[1]
        if self._acc is None:
            self._acc = {"n": torch.zeros(K, dtype=torch.float64, device=z.device),
                         "s": torch.zeros(K, D, dtype=torch.float64, device=z.device),
                         "S": torch.zeros(D, D, dtype=torch.float64, device=z.device)}
        a = self._acc
        a["n"].index_add_(0, y, torch.ones_like(y, dtype=torch.float64))
        a["s"].index_add_(0, y, z)
        a["S"] += z.T @ z

    @torch.no_grad()
    def density_end(self):
        self._collect = False
        a, self._acc = self._acc, None
        if a is None:
            return
        present = a["n"] > 0
        n, s = a["n"][present], a["s"][present]
        N, D = float(n.sum()), s.shape[1]
        mu = s / n[:, None]
        cov = (a["S"] - (mu.T * n) @ mu) / N                       # shared within-type covariance
        cov = cov + (1e-3 * cov.diagonal().mean() + 1e-8) * torch.eye(D, dtype=cov.dtype, device=cov.device)
        L = torch.linalg.cholesky(cov)
        A = torch.linalg.solve_triangular(L, torch.eye(D, dtype=cov.dtype, device=cov.device), upper=False)
        cls = torch.nonzero(present).squeeze(1)
        idx = torch.full((self.edge_type_dim,), -1, dtype=torch.long, device=cov.device)
        idx[cls] = torch.arange(len(cls), device=cov.device)
        self._dens = {"A": A, "mut": mu @ A.T, "logpi": torch.log(n / N), "idx": idx,
                      "const": 0.5 * D * float(np.log(2 * np.pi)) + float(torch.log(L.diagonal()).sum())}

    @torch.no_grad()
    def _novelty(self, z, y):
        d = self._dens
        zt = z.detach().double() @ d["A"].T
        m2 = torch.cdist(zt, d["mut"]) ** 2                          # squared Mahalanobis per type
        logp = torch.logsumexp(d["logpi"] - 0.5 * m2, dim=1) - d["const"]
        j = d["idx"][y]
        m2_y = torch.where(j >= 0, m2.gather(1, j.clamp(min=0)[:, None]).squeeze(1),
                           m2.min(1).values)
        return (-logp).float(), (0.5 * m2_y).float()

    def _abstain_forward(self, h, y, inference, class_weights):
        o = float(self.abstain.get("o", 2.0))
        if self.log_o is not None:
            o = float(self.log_o.exp())
        logp = torch.log_softmax(h, dim=-1)                 # K real types + abstain
        lp_y = logp.gather(1, y[:, None]).squeeze(1)
        lp_m = logp[:, -1]
        # distribution over the real edge types alone (abstain removed), computed stably
        lq_y = torch.log_softmax(h[:, :-1], dim=-1).gather(1, y[:, None]).squeeze(1)
        log_keep = torch.log1p(-lp_m.exp().clamp(max=1 - 1e-6))
        if inference:
            # anomaly score = surprise among the real edge types, abstain mass removed
            return -lq_y, logp, lp_m, log_keep
        self._train_steps += 1
        if self.log_o is not None:
            with torch.no_grad():
                rate = (lp_m.exp() > 0.5).float().mean()
                target = 1.0 - float(self.abstain["cov"])
                k = float(self.abstain.get("k", 0.05))
                self.log_o.add_(k * (rate - target)).clamp_(
                    float(np.log(1.05)), float(np.log(self.edge_type_dim)))
                if self._train_steps % 500 == 1:
                    print(f"[abstain] step {self._train_steps}: rate {float(rate):.3f} "
                          f"target {target:.3f} o {float(self.log_o.exp()):.2f}", flush=True)
        if not self._engaged:
            with torch.no_grad():
                m = float((-lq_y).mean())
                self._warm_ema = m if self._warm_ema is None else 0.99 * self._warm_ema + 0.01 * m
                if self._train_steps >= 100 and self._warm_ema < float(np.log(o)):
                    self._engaged = True
                    print(f"[abstain] engaged at training step {self._train_steps}: running CE "
                          f"{self._warm_ema:.3f} < log o = {np.log(o):.3f}", flush=True)
            per = -lq_y
        elif self.abstain.get("warm") != "auto" and self._train_steps <= int(self.abstain.get("warm", 0)):
            per = -lq_y
        else:
            per = -torch.logaddexp(lp_y, lp_m - torch.log(torch.tensor(o, device=h.device)))
            # "ce": keep the real-type distribution trained where the model abstains, so its
            # surprise stays meaningful there (otherwise the real logits drift freely)
            ce = float(self.abstain.get("ce", 0.0))
            if ce > 0:
                per = per + ce * (-lq_y)
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

        nov = None
        if self.density and self._z is not None and not self.multi_edge:
            if self._collect:
                self._accumulate(self._z, edge_type)
            elif inference and self._dens is not None:
                nov = self._novelty(self._z, edge_type)
            self._z = None

        if self.abstain and not self.multi_edge:
            loss, logp, lp_m, log_keep = self._abstain_forward(h, edge_type, inference, class_weights)
            out = {"loss": loss}
            if inference:
                with torch.no_grad():
                    lq = torch.log_softmax(h[:, :-1], dim=-1)   # over the real types only
                    out["entropy"] = -(lq.exp() * lq).sum(-1)
                    out["abstain"] = lp_m.exp()
                if nov is not None:
                    out["novelty"], out["novelty_y"] = nov
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
            if nov is not None:
                out["novelty"], out["novelty_y"] = nov
        return out
