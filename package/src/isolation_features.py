"""Third-player folds following aggression by a pair member on the same street."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine import CACHE, A_FOLD, A_BET, A_RAISE, A_ALLIN

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "features"
NP_PLAYERS = 12000

NAMES = ["q3f_thirdfold_share", "q3f_thirdfold_pre", "q3f_thirdfold_afterpair",
         "q3f_thirdfold_potbb", "q3f_pairin_before_fold_bb", "q3f_thirdfold_open",
         "q3f_thirdfold_afterthird", "q3f_thirdfold_afterboth", "q3f_firstaggr_pair",
         "q3f_thirdfold_n"]

_C = None


def _cache():
    global _C
    if _C is None:
        g = lambda n: np.load(CACHE / f"{n}.npy", mmap_mode="r")
        _C = dict(off=np.asarray(g("act_offsets")), seat=np.asarray(g("act_seat")),
                  act=np.asarray(g("act_action")), street=np.asarray(g("act_street")),
                  amt=np.asarray(g("act_amount")), tc=np.asarray(g("act_to_call")),
                  pot=np.asarray(g("act_pot_before")), bb=np.asarray(g("hand_bb")),
                  folded=np.asarray(g("seat_folded")), sp=np.asarray(g("seat_player")))
    return _C


def _row(C, h, sA, sB):
    a, b = int(C["off"][h]), int(C["off"][h + 1])
    seat = C["seat"][a:b].astype(np.int64)
    act = C["act"][a:b]
    street = C["street"][a:b].astype(np.int64)
    amt = C["amt"][a:b]
    tc = C["tc"][a:b]
    pot = C["pot"][a:b]
    aggr = (act == A_BET) | (act == A_RAISE) | ((act == A_ALLIN) & (amt > tc))
    bb = max(float(C["bb"][h]), 1e-9)
    n = b - a
    ispair = (seat == sA) | (seat == sB)
    share = pre = afterpair = potbb = pairin = op = afterthird = afterboth = 0.0
    firstaggr_pair = 0.0
    nfold = 0.0
    fa = np.flatnonzero(aggr)
    if len(fa):
        firstaggr_pair = float(bool(ispair[fa[0]]))
    pa_n, pa_seats, th_n = {}, {}, {}
    pair_in_bb = 0.0
    for i in range(n):
        st = int(street[i])
        if act[i] == A_FOLD and not ispair[i]:
            nfold += 1.0
            if pa_n.get(st, 0) > 0:
                afterpair += 1.0
                potbb += float(pot[i]) / bb
                pairin += pair_in_bb / bb
                if st == 0:
                    pre += 1.0
                if len(pa_seats.get(st, ())) >= 2:
                    afterboth += 1.0
                if th_n.get(st, 0) == 0:
                    op += 1.0
            elif th_n.get(st, 0) > 0:
                afterthird += 1.0
        if aggr[i]:
            if ispair[i]:
                pa_n[st] = pa_n.get(st, 0) + 1
                pa_seats.setdefault(st, set()).add(int(seat[i]))
                pair_in_bb += float(amt[i])
            else:
                th_n[st] = th_n.get(st, 0) + 1
    share = afterpair / max(nfold, 1.0)
    return (share, pre, afterpair, potbb, pairin, op, afterthird, afterboth,
            firstaggr_pair, nfold)


def compute(key: np.ndarray, hidx: np.ndarray):
    C = _cache()
    plo = (np.asarray(key) // NP_PLAYERS).astype(np.int64)
    phi = (np.asarray(key) % NP_PLAYERS).astype(np.int64)
    hh = np.asarray(hidx).astype(np.int64)
    X = np.zeros((len(hh), len(NAMES)), np.float32)
    sp = C["sp"]
    for i in range(len(hh)):
        h = int(hh[i])
        row = sp[h]
        sA = int(np.flatnonzero(row == plo[i])[0])
        sB = int(np.flatnonzero(row == phi[i])[0])
        X[i] = _row(C, h, sA, sB)
    return X, list(NAMES)


def stored(tag: str = "dev", mask=None):
    key = np.load(FEAT / f"ph_key_{tag}.npy")
    hidx = np.load(FEAT / f"ph_hidx_{tag}.npy")
    if mask is not None:
        key, hidx = key[mask], hidx[mask]
    return compute(key, hidx)[0], list(NAMES)


def live(key: np.ndarray, hidx: np.ndarray):
    return compute(key, hidx)
