"""Development hand order within each table."""
from pathlib import Path
from functools import lru_cache
import numpy as np
N_TABLES = 400
CACHE = Path(__file__).resolve().parents[1] / "cache"

@lru_cache(maxsize=1)
def _hand_order():
    """Stable timestamp order of development hands within each table; other phases return -1."""
    ht = np.load(CACHE / "hand_table.npy")
    hp = np.load(CACHE / "hand_phase.npy")
    st = np.load(CACHE / "hand_started_at.npy")
    rank = np.full(len(ht), -1, dtype=np.int32)
    for t in range(N_TABLES):
        sel = np.flatnonzero((ht == t) & (hp == 0))
        sel = sel[np.argsort(st[sel], kind="stable")]
        rank[sel] = np.arange(len(sel), dtype=np.int32)
    return rank
