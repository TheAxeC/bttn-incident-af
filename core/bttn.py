"""Block-Term Tensor Network (BTTN) for interpretable incident-AF prediction.

The campaign showed a plain (multi)linear block-term regressor cannot match a CNN:
it has no nonlinearity to exploit rich representations. BTTN keeps the block-term
STRUCTURE (and hence the glass-box maps) but trains it end-to-end with nonlinearity
and a survival objective, so it can reach CNN-level accuracy WHILE remaining
interpretable -- the genuine algorithm-level contribution of the follow-up.

Architecture (input X: B x F x T x L spectral tensor):
  - Block-term layer: K blocks, each a learnable rank-(rf,rt,rl) Tucker subspace with
    mode factors A_k (F x rf), B_k (T x rt), C_k (L x rl). Block k projects X onto its
    subspace -> small core Z_k = X x1 A_k^T x2 B_k^T x3 C_k^T (rf x rt x rl features).
    The factors A_k/B_k/C_k ARE the interpretable per-block frequency / time / lead
    signatures (rank-structured (F x T x L) glass-box maps).
  - Nonlinear head: BatchNorm -> MLP(ReLU,Dropout) -> scalar log-risk. This is the
    capacity the plain block regressor lacked; the predictive path still runs THROUGH
    the block subspaces, so the maps stay faithful.
  - Loss: Cox partial likelihood (within-batch risk sets) -> proper time-to-event fit
    with right-censoring (controls censored at follow-up).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

DEV = torch.device("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
MAX_EPOCH, PATIENCE, BATCH = 120, 16, 512


class BlockTermLayer(nn.Module):
    """Convolutional block-term layer. Each block k projects the freq and lead modes
    onto learned factors A_k (freq signature) and C_k (lead signature), then applies
    LOCAL temporal conv kernels along time (capturing f-wave bursts / rhythm timing the
    global Tucker projection missed). Glass-box: A_k = per-block frequency profile,
    C_k = per-block lead/spatial profile, conv kernels = local temporal templates."""

    def __init__(self, F, T, L, K=12, rf=3, rl=2, ksizes=(3, 7, 15), ct=6, temporal="conv",
                 block_mode="blockterm"):
        super().__init__()
        # block_mode (block-term-structure ablation) controls the
        # interpretable first-layer parameterisation; everything downstream is identical:
        #   'blockterm' : K blocks, each a learnable rank-(rf,rl) Tucker freq/lead subspace
        #                 (A_k in R^{F x rf}, C_k in R^{L x rl}) -- the contribution.
        #   'cp'        : CP rank-1-per-block (rf=rl=1) -> K rank-1 freq(x)lead atoms; tests
        #                 whether the multilinear RANK (>1) inside a block does work.
        #   'fullrank'  : a single full-rank learned F x L -> ch linear projection per time
        #                 step (no block / no multilinear-rank structure) -> tests whether the
        #                 block-term parameterisation itself beats plain dim-reduction.
        if block_mode == "cp":
            rf, rl = 1, 1
        self.block_mode = block_mode
        self.K, self.rf, self.rl, self.ct, self.temporal = K, rf, rl, ct, temporal
        ch = K * rf * rl
        if block_mode == "fullrank":
            # one dense F*L -> ch projection (same output width as the K-block variant),
            # applied per time step: no block / no per-mode low-rank factors.
            self.W = nn.Parameter(torch.randn(ch, F, L) * (1.0 / (F * L) ** 0.5))
        else:
            self.A = nn.Parameter(torch.randn(K, F, rf) * 0.1)
            self.C = nn.Parameter(torch.randn(K, L, rl) * 0.1)
        self.ch_out = ch * ct * len(ksizes)
        # multi-scale grouped temporal convs: fast (f-waves) + slow (rhythm) templates
        self.tconvs = nn.ModuleList([
            nn.Conv1d(ch, ch * ct, ks, padding=ks // 2, groups=ch) for ks in ksizes])
        self.bn = nn.BatchNorm1d(self.ch_out)
        self.attn = nn.Conv1d(self.ch_out, 1, 1)   # temporal attention (interpretable: when)
        if temporal == "gru":                      # recurrent over the spectrogram time axis
            self.gru = nn.GRU(ch, ch * ct, batch_first=True, bidirectional=True)
            self._gru_out = 2 * (ch * ct) * 2      # bi-directional * (mean+max)

    def _project(self, x):                         # (B,F,T,L) -> (B, ch, T) pure multilinear
        if self.block_mode == "fullrank":          # dense F x L -> ch projection per time step
            return torch.einsum("bftl,cfl->bct", x, self.W)
        z = torch.einsum("bftl,kfp->bkptl", x, self.A)
        z = torch.einsum("bkptl,kls->bkpts", z, self.C)
        B_, K, rf, T, rl = z.shape
        return z.permute(0, 1, 2, 4, 3).reshape(B_, K * rf * rl, T)

    def block_scores(self, x):                     # interpretable features: pure block projection, time-pooled
        return self._project(x).mean(-1)           # (B, K*rf*rl)

    def forward(self, x, tokens=False):            # x: (B, F, T, L)
        z0 = self._project(x)                       # (B, ch, T)
        if self.temporal == "gru":
            g, _ = self.gru(z0.transpose(1, 2))     # (B, T, 2*ch*ct)
            return torch.cat([g.mean(1), g.amax(1)], dim=1)        # (B, _gru_out)
        z = torch.relu(self.bn(torch.cat([c(z0) for c in self.tconvs], dim=1)))  # (B, ch_out, T)
        a = torch.softmax(self.attn(z), dim=-1)
        z_att = (z * a).sum(-1); z_max = z.amax(-1)
        if tokens:
            nk, per = len(self.tconvs), self.rf * self.rl * self.ct
            tok = lambda v: v.view(z.shape[0], nk, self.K, per).permute(0, 2, 1, 3).reshape(z.shape[0], self.K, nk * per)
            return torch.cat([tok(z_att), tok(z_max)], dim=2)
        return torch.cat([z_att, z_max], dim=1)

    def out_dim(self):
        return self._gru_out if self.temporal == "gru" else 2 * self.ch_out

    def token_dim(self):
        return 2 * len(self.tconvs) * self.rf * self.rl * self.ct

    def maps(self):
        """K rank-structured (F, L) freq x lead glass-box signatures per block (the
        temporal templates are the conv kernels, reported separately)."""
        if self.block_mode == "fullrank":                # dense channels -> (ch, F, L) atoms
            return self.W.detach().cpu().numpy()
        A, C = self.A.detach(), self.C.detach()
        M = torch.einsum("kfp,klp->kfl", A[..., :min(self.rf, self.rl)],
                         C[..., :min(self.rf, self.rl)])
        return M.cpu().numpy()                            # (K, F, L)


class BTTN(nn.Module):
    """head_type: 'mlp' (default) or 'attn' (transformer over per-block tokens).
    aux: if True, add a second head for the multi-task AF-detection auxiliary task."""

    def __init__(self, F, T, L, K=12, rf=3, rl=2, hidden=256, p=0.4, head_type="mlp", aux=False,
                 temporal="conv", deep_head=False, block_mode="blockterm", cov_dim=0):
        super().__init__()
        self.block = BlockTermLayer(F, T, L, K, rf, rl, temporal=temporal, block_mode=block_mode)
        self.head_type = head_type
        # Optional demographic covariate term (age, sex): a zero-initialised LINEAR map added to the
        # head log-risk. It bypasses the block layer entirely, so the A_k / C_k glass-box signatures
        # and the block-score faithfulness probe are unchanged -- demographics at no interpretability
        # cost (the ECG evidence still routes through the interpretable blocks; age/sex is a separate,
        # readable additive adjustment). Off by default (cov_dim=0) so plain BTTN is untouched.
        self.cov = nn.Linear(cov_dim, 1, bias=False) if cov_dim else None
        if self.cov is not None:
            nn.init.zeros_(self.cov.weight)
        if head_type == "attn":
            dt = self.block.token_dim(); dm = 96
            self.proj = nn.Linear(dt, dm)
            enc = nn.TransformerEncoderLayer(dm, nhead=4, dim_feedforward=2 * dm,
                                             dropout=p, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, num_layers=2)
            d = dm
        else:
            d = self.block.out_dim()
        if deep_head:                                # wider/deeper nonlinear head (capacity test)
            self.head = nn.Sequential(
                nn.BatchNorm1d(d), nn.Dropout(p),
                nn.Linear(d, 2 * hidden), nn.ReLU(), nn.BatchNorm1d(2 * hidden), nn.Dropout(p),
                nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(p),
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden // 2, 1))
        else:
            self.head = nn.Sequential(
                nn.BatchNorm1d(d), nn.Dropout(p),
                nn.Linear(d, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(p),
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p),
                nn.Linear(hidden // 2, 1))
        self.aux_head = nn.Sequential(nn.Linear(d, hidden // 2), nn.ReLU(),
                                      nn.Dropout(p), nn.Linear(hidden // 2, 1)) if aux else None

    def block_scores(self, x):
        if x.ndim == 3:
            x = x[:, None]
        return self.block.block_scores(x)            # (B, K*rf*rl) interpretable features

    def features(self, x):
        if self.head_type == "attn":
            t = self.tr(self.proj(self.block(x, tokens=True)))   # (B, K, dm)
            return t.mean(1)                                     # pool over blocks
        return self.block(x)

    def forward(self, x, cov=None, aux=False):
        f = self.features(x)
        if aux and self.aux_head is not None:
            return self.aux_head(f).squeeze(-1)
        out = self.head(f).squeeze(-1)
        if self.cov is not None and cov is not None:
            out = out + self.cov(cov).squeeze(-1)
        return out


def cox_ph_loss(risk, time, event):
    """Negative Cox partial log-likelihood with within-batch risk sets."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cum = torch.logcumsumexp(risk, dim=0)          # risk set = all with time >= t_i
    pl = (risk - log_cum) * event
    return -pl.sum() / (event.sum() + 1e-8)


def _std(Xtr, Xte):
    m = Xtr.mean((0, 1, 2), keepdims=True); s = Xtr.std((0, 1, 2), keepdims=True) + 1e-6
    return (Xtr - m) / s, (Xte - m) / s


def _std_cov(cov_tr, cov_te):
    """Standardise demographic covariates using TRAIN stats only (leakage-free); impute NaN
    (e.g. missing anchor_age) with the train mean, which is 0 after centering."""
    m = np.nanmean(cov_tr, axis=0); s = np.nanstd(cov_tr, axis=0) + 1e-6
    z = lambda c: ((np.where(np.isnan(c), m, c) - m) / s).astype(np.float32)
    return z(cov_tr), z(cov_te)


def bttn_fold(Xtr, time_tr, event_tr, Xte, y_tr_horizon, seed=0, n_ens=3, pretrained=None,
              head_type="mlp", K=12, ace_warm=False, block_mode="blockterm",
              cov_tr=None, cov_te=None):
    """3-seed ensemble (fair vs the CNN baseline). Returns averaged test risk.
    pretrained: SSL-pretrained BlockTermLayer to warm-start; ace_warm: ACE supervised init.
    head_type: 'mlp' or 'attn'; K: number of block terms (ablation knob);
    block_mode: 'blockterm'|'cp'|'fullrank' (first-layer-structure ablation);
    cov_tr/cov_te: optional demographic covariates (age, sex) -> linear head term."""
    if Xtr.ndim == 3:
        Xtr, Xte = Xtr[:, None], Xte[:, None]
    Xtr, Xte = _std(Xtr.astype(np.float32), Xte.astype(np.float32))
    covt = covte = None
    if cov_tr is not None:
        covt, covte = _std_cov(np.asarray(cov_tr, float), np.asarray(cov_te, float))
    ens = np.zeros(len(Xte))
    for sd in range(n_ens):
        ens += _bttn_train_one(Xtr, time_tr, event_tr, Xte, y_tr_horizon, seed * 10 + sd,
                               pretrained, head_type, K, ace_warm, block_mode, covt, covte)
    return ens / n_ens


def _bttn_train_one(Xtr, time_tr, event_tr, Xte, y_tr_horizon, seed, pretrained=None,
                    head_type="mlp", K=12, ace_warm=False, block_mode="blockterm",
                    cov_tr=None, cov_te=None):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xtr)); cut = int(0.85 * len(idx))
    tri, vai = idx[:cut], idx[cut:]
    F, T, L = Xtr.shape[1:]
    cov_dim = 0 if cov_tr is None else cov_tr.shape[1]
    model = BTTN(F, T, L, K=K, head_type=head_type, block_mode=block_mode, cov_dim=cov_dim).to(DEV)
    if ace_warm:                                          # ACE supervised cross-cov init
        A, Cc = ace_init(Xtr[tri], y_tr_horizon[tri], model.block.K, model.block.rf, model.block.rl, seed)
        model.block.A.data = torch.tensor(A, device=DEV); model.block.C.data = torch.tensor(Cc, device=DEV)
    if pretrained is not None:
        sd = torch.load(pretrained, map_location=DEV)
        model.block.load_state_dict(sd, strict=False)     # warm-start from SSL encoder
    if model.cov is not None and cov_tr is not None:      # warm-start covariates to clinical strength
        from sklearn.linear_model import LogisticRegression
        try:
            lr_w = LogisticRegression(max_iter=1000, class_weight="balanced").fit(cov_tr[tri], y_tr_horizon[tri])
            with torch.no_grad():
                model.cov.weight.copy_(torch.tensor(lr_w.coef_.astype(np.float32), device=DEV))
        except Exception:
            pass
    # keep the (warm-started) covariate term OUT of weight decay so it is not eroded to zero
    cov_ids = {id(p) for p in (model.cov.parameters() if model.cov is not None else [])}
    param_groups = [{"params": [p for p in model.parameters() if id(p) not in cov_ids]}]
    if model.cov is not None:
        param_groups.append({"params": list(model.cov.parameters()), "weight_decay": 0.0})
    opt = torch.optim.Adam(param_groups, lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCH)
    Xt = torch.tensor(Xtr[tri], device=DEV); tt = torch.tensor(time_tr[tri], dtype=torch.float32, device=DEV)
    et = torch.tensor(event_tr[tri], dtype=torch.float32, device=DEV)
    Xv = torch.tensor(Xtr[vai], device=DEV); yv = y_tr_horizon[vai]
    Xe = torch.tensor(Xte, device=DEV)
    covt = torch.tensor(cov_tr[tri], device=DEV) if cov_tr is not None else None
    covv = torch.tensor(cov_tr[vai], device=DEV) if cov_tr is not None else None
    cove = torch.tensor(cov_te, device=DEV) if cov_te is not None else None
    n = len(Xt); best_auc, best_state, bad = -1, None, 0
    for ep in range(MAX_EPOCH):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            if et[b].sum() < 1:
                continue
            opt.zero_grad()
            loss = cox_ph_loss(model(Xt[b], cov=covt[b] if covt is not None else None),
                               tt[b], et[b]); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va = model(Xv, cov=covv).cpu().numpy()
        auc = roc_auc_score(yv, va) if len(np.unique(yv)) == 2 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return model(Xe, cov=cove).cpu().numpy()


# --- reusable train/predict (for interpretability + longitudinal analysis) ----- #
def std_stats(X):
    m = X.mean((0, 1, 2), keepdims=True); s = X.std((0, 1, 2), keepdims=True) + 1e-6
    return m, s


def ace_init(X, y, K, rf, rl, seed=0):
    """ACE-style supervised warm-start for the block factors. ACE decomposes the X-Y
    cross-covariance tensor; here we form C = sum_n (y_n-ybar) X_n (F x T x L), take the
    dominant FREQUENCY and LEAD directions (SVD of the mode unfoldings), and use them to
    initialize A_k (freq) and C_k (lead) for all K blocks (+ small noise to break symmetry).
    The blocks thus START in the supervised covariance subspace, then fine-tune by gradient."""
    X = np.asarray(X, np.float32)
    w = (np.asarray(y, np.float32) - np.mean(y))
    C = np.einsum("n,nftl->ftl", w, X)                       # supervised cross-cov (F,T,L)
    F, T, L = C.shape
    Uf = np.linalg.svd(C.reshape(F, T * L), full_matrices=False)[0][:, :rf]          # (F, rf)
    Ul = np.linalg.svd(C.transpose(2, 0, 1).reshape(L, F * T), full_matrices=False)[0][:, :rl]  # (L, rl)
    rng = np.random.RandomState(seed)
    A = np.stack([Uf + 0.02 * rng.randn(F, rf) for _ in range(K)]).astype(np.float32)
    Cc = np.stack([Ul + 0.02 * rng.randn(L, rl) for _ in range(K)]).astype(np.float32)
    return A, Cc


def train_model(X, time, event, y_es, seed=0, pretrained=None, head_type="mlp",
                temporal="conv", deep_head=False, block_mode="blockterm"):
    """Train one BTTN on all of X; return (model, (mean,std)) for later inference."""
    torch.manual_seed(seed)
    if X.ndim == 3:
        X = X[:, None]
    X = X.astype(np.float32); m, s = std_stats(X); Xs = (X - m) / s
    rng = np.random.RandomState(seed); idx = rng.permutation(len(Xs)); cut = int(0.85 * len(idx))
    tri, vai = idx[:cut], idx[cut:]
    F, T, L = Xs.shape[1:]
    model = BTTN(F, T, L, head_type=head_type, temporal=temporal, deep_head=deep_head,
                 block_mode=block_mode).to(DEV)
    if pretrained is not None:
        model.block.load_state_dict(torch.load(pretrained, map_location=DEV), strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCH)
    Xt = torch.tensor(Xs[tri], device=DEV); tt = torch.tensor(time[tri], dtype=torch.float32, device=DEV)
    et = torch.tensor(event[tri], dtype=torch.float32, device=DEV)
    Xv = torch.tensor(Xs[vai], device=DEV); yv = y_es[vai]
    n = len(Xt); best_auc, best_state, bad = -1, None, 0
    for ep in range(MAX_EPOCH):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            if et[b].sum() < 1:
                continue
            opt.zero_grad(); cox_ph_loss(model(Xt[b]), tt[b], et[b]).backward(); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            va = model(Xv).cpu().numpy()
        auc = roc_auc_score(yv, va) if len(np.unique(yv)) == 2 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model, (m, s)


def train_model_binary(X, y, seed=0, head_type="mlp", K=12, pretrained=None, epochs=80):
    """Binary (BCE) BTTN fit - for tasks without time-to-event (e.g. PTB-XL AF detection /
    AF-in-SR identification, used for external validation). Same architecture; logistic loss."""
    torch.manual_seed(seed)
    if X.ndim == 3:
        X = X[:, None]
    X = X.astype(np.float32); m, s = std_stats(X); Xs = (X - m) / s
    rng = np.random.RandomState(seed); idx = rng.permutation(len(Xs)); cut = int(0.85 * len(idx))
    tri, vai = idx[:cut], idx[cut:]
    model = BTTN(*Xs.shape[1:], K=K, head_type=head_type).to(DEV)
    if pretrained is not None:
        model.block.load_state_dict(torch.load(pretrained, map_location=DEV), strict=False)
    posw = torch.tensor([(y[tri] == 0).sum() / max(1, (y[tri] == 1).sum())], dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xs[tri], device=DEV); yt = torch.tensor(y[tri].astype(np.float32), device=DEV)
    Xv = torch.tensor(Xs[vai], device=DEV); yv = y[vai]
    best, bestst, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); lossf(model(Xt[b]), yt[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = model(Xv).cpu().numpy()
        au = roc_auc_score(yv, va) if len(np.unique(yv)) == 2 else 0.5
        if au > best + 1e-4:
            best, bestst, bad = au, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12:
                break
    model.load_state_dict(bestst); model.eval()
    return model, (m, s)


def predict_risk(model, X, stats):
    m, s = stats
    if X.ndim == 3:
        X = X[:, None]
    Xs = (X.astype(np.float32) - m) / s
    with torch.no_grad():
        return model(torch.tensor(Xs, device=DEV)).cpu().numpy()
