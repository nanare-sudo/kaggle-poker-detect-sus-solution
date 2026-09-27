"""Paar-Universum je Phase: alle gemeinsam gespielten Spielerpaare mit Zaehlern."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
NP = 12000
COMBOS = list(combinations(range(6), 2))


def load(name):
    return np.load(CACHE / f"{name}.npy", allow_pickle=False, mmap_mode=None)


def pair_hand_index(phase: int | None = None):
    """Liefert (key, hidx) fuer jede (Paar, Hand)-Kombination der gewaehlten Phase.

    key = plo * NP + phi mit plo < phi (Spielercodes). 15 Paare je Hand.
    """
    sp = load("seat_player").astype(np.int32)          # (H,6)
    ph = load("hand_phase")
    sel = np.arange(len(sp), dtype=np.int32) if phase is None else np.flatnonzero(ph == phase).astype(np.int32)
    sp = sp[sel]
    keys = np.empty((len(sel), 15), dtype=np.int64)
    for c, (i, j) in enumerate(COMBOS):
        a, b = sp[:, i], sp[:, j]
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b).astype(np.int64)
        keys[:, c] = lo * NP + hi
    hidx = np.repeat(sel, 15)
    return keys.ravel(), hidx, sel


def build_universe(phase: int, min_shared: int = 0):
    key, hidx, _ = pair_hand_index(phase)
    order = np.argsort(key, kind="stable")
    key_s, hidx_s = key[order], hidx[order]
    uniq, start, cnt = np.unique(key_s, return_index=True, return_counts=True)
    if min_shared > 0:
        m = cnt >= min_shared
        uniq, start, cnt = uniq[m], start[m], cnt[m]
    return uniq, cnt, key_s, hidx_s, start
