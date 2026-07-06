"""Lazy waveform access for MIMIC-IV-ECG.

The 36 GB matched subset stays zipped; we pull individual records (.hea + .dat)
out of the archive on demand and cache them under config.WAVE_CACHE. wfdb then reads
the cached pair. Returns a config.Recording(signal=(T,12), fs).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import wfdb

from core import config

_ZIP_ROOT = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
_zf = None


def _zip():
    global _zf
    if _zf is None:
        _zf = zipfile.ZipFile(config.MIMIC_ECG_ZIP, "r")
    return _zf


def _ensure_local(rel_path):
    """rel_path e.g. files/p1000/p10000032/s40689238/40689238 -> local base path (no ext)."""
    base = config.WAVE_CACHE.parent / rel_path           # ~/data/mimic-ecg/files/...
    hea, dat = base.with_suffix(".hea"), base.with_suffix(".dat")
    if hea.exists() and dat.exists():
        return base
    base.parent.mkdir(parents=True, exist_ok=True)
    zf = _zip()
    for ext, dst in ((".hea", hea), (".dat", dat)):
        member = f"{_ZIP_ROOT}/{rel_path}{ext}"
        with zf.open(member) as src, open(dst, "wb") as out:
            out.write(src.read())
    return base


def load(rel_path):
    base = _ensure_local(rel_path)
    rec = wfdb.rdrecord(str(base))
    sig = np.asarray(rec.p_signal, dtype=float)          # (T, 12)
    # MIMIC lead order matches config.LEAD_NAMES; guard against NaNs from lead dropout
    sig = np.nan_to_num(sig, nan=0.0)
    return config.Recording(signal=sig, fs=int(rec.fs))


def load_many(rel_paths):
    out = []
    for p in rel_paths:
        try:
            out.append(load(p))
        except Exception as e:                            # missing/corrupt record
            out.append(None)
    return out


if __name__ == "__main__":
    r = load("files/p1000/p10000032/s40689238/40689238")
    print("signal", r.signal.shape, "fs", r.fs, "range",
          round(float(r.signal.min()), 2), round(float(r.signal.max()), 2))
