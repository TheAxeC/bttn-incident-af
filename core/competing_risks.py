"""Discrete-time competing-risks (DeepHit-style) loss for incident-AF.

Many elderly MIMIC patients DIE before developing AF. Treating death as ordinary
censoring overstates AF risk; competing-risks models it as a separate absorbing
event. The BTTN survival head is swapped for a discrete-time hazard over K bins x
2 causes (AF, death); this loss is the negative log-likelihood of that distribution.
Activates on gold data (needs death dates `dod`); validated here on synthetic data.

Output convention: logits f of shape (N, n_causes, n_bins). p = softmax over all
(cause,bin) cells; survival-to-bin-k = 1 - cumulative p up to k.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def to_bins(time, n_bins, tmax):
    e = np.clip((np.asarray(time, float) / tmax * n_bins).astype(int), 0, n_bins - 1)
    return e


def deephit_loss(logits, bin_idx, cause, n_bins, n_causes=2, alpha=0.2, sigma=0.1):
    """logits (N, n_causes, n_bins); cause: 0=censored, 1..n_causes = event cause.
    L = NLL (event: prob mass in its cell; censored: prob of surviving past its bin)
        + alpha * cause-specific ranking loss (concordance surrogate)."""
    N = logits.shape[0]
    p = torch.softmax(logits.reshape(N, -1), dim=1).reshape(N, n_causes, n_bins)  # joint over (cause,bin)
    cif = torch.cumsum(p.sum(1), dim=1)                       # P(any event by bin k)  (N, n_bins)
    dev = logits.device
    ev = cause > 0
    nll = torch.zeros(N, device=dev)
    # event subjects: -log p[i, cause-1, bin]
    if ev.any():
        idx = torch.arange(N, device=dev)[ev]
        nll[ev] = -torch.log(p[idx, cause[ev] - 1, bin_idx[ev]] + 1e-8)
    # censored: -log P(no event through their bin) = -log(1 - cif[i, bin])
    if (~ev).any():
        idx = torch.arange(N, device=dev)[~ev]
        surv = torch.clamp(1.0 - cif[idx, bin_idx[~ev]], min=1e-6)
        nll[~ev] = -torch.log(surv)
    loss = nll.mean()
    # ranking loss for the event of interest (cause 1 = AF): earlier-AF should have higher CIF
    af = cause == 1
    if af.sum() > 1:
        cif_af = torch.cumsum(p[:, 0, :], dim=1)             # (N, n_bins)
        ai = torch.arange(N, device=dev)[af]
        ri = cif_af[ai, bin_idx[af]]                         # risk at own bin
        # pairs (i AF at k_i) vs (j still at risk at k_i): risk_i should exceed risk_j
        rank = 0.0; cnt = 0
        for a in range(len(ai)):
            i = ai[a]; ki = bin_idx[i]
            others = bin_idx > ki
            if others.any():
                diff = torch.clamp(cif_af[i, ki] - cif_af[others, ki], -1.0, 1.0)
                rank = rank + torch.exp(-diff / sigma).mean(); cnt += 1
        if cnt:
            loss = loss + alpha * (rank / cnt)
    return loss


# --------------------------- synthetic self-test ------------------------------ #
if __name__ == "__main__":
    torch.manual_seed(0); rng = np.random.RandomState(0)
    N, D, K = 4000, 6, 8
    X = rng.randn(N, D).astype(np.float32)
    risk_af = X @ rng.randn(D)                                # AF time depends on X
    t_af = np.maximum(0.2, 6 * np.exp(-0.6 * (risk_af - risk_af.mean()) / risk_af.std()) + rng.rand(N))
    t_death = rng.exponential(8, N)                          # competing death, independent
    t_cens = rng.uniform(2, 10, N)
    t = np.minimum(np.minimum(t_af, t_death), t_cens)
    cause = np.where((t_af <= t_death) & (t_af <= t_cens), 1,
                     np.where((t_death < t_af) & (t_death <= t_cens), 2, 0))
    tmax = np.quantile(t, 0.99); b = to_bins(t, K, tmax)
    Xt = torch.tensor(X); bt = torch.tensor(b); ct = torch.tensor(cause)
    net = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2 * K))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    tr = np.arange(N) % 5 != 0; te = ~tr
    for ep in range(200):
        net.train(); opt.zero_grad()
        lg = net(Xt[tr]).reshape((-1, 2, K))
        loss = deephit_loss(lg, bt[tr], ct[tr], K)
        loss.backward(); opt.step()
    # evaluate AF-cause discrimination (C-index for cause 1) on test
    from lifelines.utils import concordance_index
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(Xt[te]).reshape((-1, 2, K)).reshape(te.sum(), -1), 1).reshape(te.sum(), 2, K)
        cif_af = torch.cumsum(p[:, 0, :], 1)[:, -1].numpy()  # total AF cumulative incidence
    af_event = (cause[te] == 1).astype(int)
    # among those with AF or censored (death is competing -> treat as censored for AF C-index)
    ci = concordance_index(t[te], -cif_af, af_event)
    print(f"[competing-risks synthetic] final loss {loss.item():.3f}  AF cause C-index {ci:.3f}  "
          f"(>0.5 means it learns AF risk under competing death)")
