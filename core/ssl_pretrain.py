"""Self-supervised contrastive pretraining of the BTTN block-term encoder.

The labeled incident-AF set is small and label-noisy; the unlabeled ECG pool is huge
(800k). We pretrain the SAME block-term layer the BTTN uses, label-free, via SimCLR-
style contrastive learning on spectral tensors, then transfer its weights to the BTTN
for fine-tuning. Encoder == BlockTermLayer, so the transfer is exact and the glass-box
structure (freq/lead factors + temporal kernels) is what gets pretrained.

Augmentations (ECG-appropriate, preserve AF-relevant structure): time masking, freq
masking, lead dropout, additive noise, small time roll.

Run:  python ssl_pretrain.py --n 30000 --epochs 60
Saves: results/ssl_encoder.pt  (BlockTermLayer state_dict)
"""
from __future__ import annotations

import argparse
import time as _time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import config
from core import data
from core.bttn import BlockTermLayer, _std, DEV


def sample_pool(n, seed=0):
    rl = pd.read_csv(config.MIMIC_ECG_DIR / "record_list.csv", usecols=["subject_id", "path"])
    rl = rl.drop_duplicates("subject_id")                 # one ECG/patient -> diverse pool
    return rl.sample(min(n, len(rl)), random_state=seed)["path"].tolist()


def build_pool_tensor(n, seed=0):
    cache = config.RESULTS_DIR / f"X_spectral_pool_n{n}.npy"
    if cache.exists():
        return np.load(cache)
    from core.preprocess import build_spectral_tensor
    paths = sample_pool(n, seed)
    t = _time.time()
    recs = data.load_many(paths)
    recs = [r for r in recs if r is not None]
    X = build_spectral_tensor(recs)
    np.save(cache, X)
    print(f"[pool] {X.shape} in {_time.time()-t:.0f}s")
    return X


def augment(x):                                           # x: (B,F,T,L) tensor on DEV
    B, Fr, T, L = x.shape
    x = x.clone()
    # time mask
    w = int(0.2 * T)
    s = torch.randint(0, max(1, T - w), (B,), device=DEV)
    for i in range(B):
        x[i, :, s[i]:s[i] + w] = 0
    # freq mask
    wf = int(0.2 * Fr); sf = torch.randint(0, max(1, Fr - wf), (B,), device=DEV)
    for i in range(B):
        x[i, sf[i]:sf[i] + wf] = 0
    # lead dropout
    mask = (torch.rand(B, 1, 1, L, device=DEV) > 0.2).float()
    x = x * mask
    # noise + small roll
    x = x + 0.05 * torch.randn_like(x)
    x = torch.roll(x, shifts=int(np.random.randint(-3, 4)), dims=2)
    return x


class Encoder(nn.Module):
    def __init__(self, Fr, T, L):
        super().__init__()
        self.block = BlockTermLayer(Fr, T, L)
        self.proj = nn.Sequential(nn.Linear(self.block.out_dim(), 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x):
        return F.normalize(self.proj(self.block(x)), dim=1)


def nt_xent(z1, z2, tau=0.2):
    z = torch.cat([z1, z2], 0); n = z1.shape[0]
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(n, device=DEV)
    targets = torch.cat([targets + n, targets])
    return F.cross_entropy(sim, targets)


def main(n, epochs, batch):
    X = build_pool_tensor(n)
    X = _std(X.astype(np.float32), X.astype(np.float32))[0]
    Xt = torch.tensor(X, device=DEV)
    Fr, T, L = X.shape[1:]
    enc = Encoder(Fr, T, L).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(Xt)
    for ep in range(epochs):
        enc.train(); perm = torch.randperm(N, device=DEV); tot = 0.0
        for i in range(0, N, batch):
            b = perm[i:i + batch]
            if len(b) < 8:
                continue
            xb = Xt[b]
            opt.zero_grad()
            loss = nt_xent(enc(augment(xb)), enc(augment(xb)))
            loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  contrastive loss {tot/(N//batch+1):.4f}")
    torch.save(enc.block.state_dict(), config.RESULTS_DIR / "ssl_encoder.pt")
    print(f"[saved] results/ssl_encoder.pt  (pretrained on {N} ECGs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    main(a.n, a.epochs, a.batch)
