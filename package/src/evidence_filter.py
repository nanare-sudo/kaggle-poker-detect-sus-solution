"""Evidence candidate filter: both pair members have a non-blind action in the hand."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUT = CACHE / "evidence_filter"
NP_PLAYERS = 12000
_ACTED = None


def build_acted(force: bool = False) -> Path:
    """(n_hands, 6) bool: hat der Sitz mindestens eine Nicht-Blind-Aktion?"""
    p = OUT / "acted.npy"
    if p.exists() and not force:
        return p
    import json
    meta = json.loads((CACHE / "meta.json").read_text())
    vocab = meta["actions_vocab"]
    blind = np.array([vocab.index(x) for x in ("post_blind", "post_ante")], dtype=np.int8)
    a_h = np.load(CACHE / "act_hidx.npy")
    a_s = np.load(CACHE / "act_seat.npy")
    a_a = np.load(CACHE / "act_action.npy")
    nb = ~np.isin(a_a, blind)
    acted = np.zeros((meta["n_hands"], 6), dtype=bool)
    acted[a_h[nb].astype(np.int64), a_s[nb]] = True
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(p, acted)
    return p


def _acted():
    global _ACTED
    if _ACTED is None:
        _ACTED = np.load(build_acted(), mmap_mode="r")
    return _ACTED


def mask(key: np.ndarray, hidx: np.ndarray) -> np.ndarray:
    """True = Hand ist als Evidenz zugelassen (beide Paarspieler handeln)."""
    key = np.asarray(key, dtype=np.int64)
    hidx = np.asarray(hidx, dtype=np.int64)
    p_lo = (key // NP_PLAYERS).astype(np.int32)
    p_hi = (key % NP_PLAYERS).astype(np.int32)
    sp_all = np.load(CACHE / "seat_player.npy", mmap_mode="r")
    ac_all = _acted()
    out = np.zeros(len(key), dtype=bool)
    CH = 400_000
    for a in range(0, len(key), CH):
        b = min(a + CH, len(key))
        h = hidx[a:b]
        sp = np.asarray(sp_all[h])
        ac = np.asarray(ac_all[h])
        out[a:b] = (((sp == p_lo[a:b, None]) & ac).any(1)
                    & ((sp == p_hi[a:b, None]) & ac).any(1))
    return out
