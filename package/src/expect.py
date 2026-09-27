"""Empirische Erwartungstabelle je Situationsklasse (einmalig, labelfrei)."""
from __future__ import annotations
from multiprocessing import Pool
import numpy as np
import engine
from engine import CACHE, A, action_bin, N_BINS


def _work(table: int):
    ht, hp, st = A("hand_table"), A("hand_phase"), A("hand_started_at")
    cnt = np.zeros((N_BINS, 5), dtype=np.float64)
    for ph in (0, 1):
        sel = np.flatnonzero((ht == table) & (hp == ph))
        sel = sel[np.argsort(st[sel], kind="stable")]
        b, cls = engine.action_bins_of(sel)
        np.add.at(cnt[:, 0], b, cls == 0)   # aggressiv
        np.add.at(cnt[:, 1], b, cls == 1)   # fold
        np.add.at(cnt[:, 2], b, cls == 2)   # call
        np.add.at(cnt[:, 3], b, cls == 3)   # check
        np.add.at(cnt[:, 4], b, 1.0)        # gesamt
    return cnt
