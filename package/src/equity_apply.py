"""Berechnet einmalig die Entscheidungs-Equity fuer alle relevanten Aktionen.

Fuer jede Aktion, die einem Einsatz gegenuebersteht (`to_call > 0`) und ein Fold
oder Call ist, wird die Equity des Handelnden gegen das TATSAECHLICHE Blatt des
letzten Aggressors bestimmt — mit dem Board, das zu diesem Zeitpunkt sichtbar
war. Ergebnis: `cache/act_equity.npy` (float32, NaN wo nicht anwendbar).
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np

import engine
import equity
from engine import A, CACHE

_EQ = None


def _init():
    global _EQ
    _EQ = equity.build_preflop_table()


def _work(table: int):
    ht, hp, st = A("hand_table"), A("hand_phase"), A("hand_started_at")
    off = A("act_offsets")
    res = []
    for ph in (0, 1):
        hsel = np.flatnonzero((ht == table) & (hp == ph))
        hsel = hsel[np.argsort(st[hsel], kind="stable")]
        H = len(hsel)
        if H == 0:
            continue
        btn = A("hand_button_seat")[hsel].astype(np.int64)
        board = A("hand_board")[hsel].astype(np.int16)
        nb = A("hand_n_board")[hsel].astype(np.int64)
        h1 = A("seat_hole1")[hsel].astype(np.int16)
        h2 = A("seat_hole2")[hsel].astype(np.int16)
        lens = off[hsel + 1] - off[hsel]
        idx = np.concatenate([np.arange(off[h], off[h + 1]) for h in hsel])
        hloc = np.repeat(np.arange(H, dtype=np.int32), lens)
        a_seat = A("act_seat")[idx].astype(np.int64)
        a_act = A("act_action")[idx].astype(np.int8)
        a_str = A("act_street")[idx].astype(np.int64)
        a_amt = A("act_amount")[idx].astype(np.float32)
        a_tc = A("act_to_call")[idx].astype(np.float32)
        bb_seat = ((btn + 2) % 6).astype(np.int8)
        aggr = engine.last_aggressor(hloc, a_str.astype(np.int8), a_seat.astype(np.int8),
                                     a_act, a_amt, a_tc, bb_seat).astype(np.int64)
        want = (a_tc > 0) & (aggr >= 0) & (aggr != a_seat) & (
            (a_act == engine.A_FOLD) | (a_act == engine.A_CALL) | (a_act == engine.A_ALLIN))
        w = np.flatnonzero(want)
        if len(w) == 0:
            continue
        hh, aa, ll = hloc[w], a_seat[w], a_str[w]
        hero = np.stack([h1[hh, aa], h2[hh, aa]], axis=1)
        vill = np.stack([h1[hh, aggr[w]], h2[hh, aggr[w]]], axis=1)
        eq = np.full(len(w), np.nan, dtype=np.float32)
        pre = ll == 0
        if pre.any():
            ch = equity.hand_class(hero[pre, 0], hero[pre, 1]).astype(np.int64)
            cv = equity.hand_class(vill[pre, 0], vill[pre, 1]).astype(np.int64)
            eq[pre] = _EQ[ch, cv]
        post = ~pre
        if post.any():
            # sichtbare Boardkarten: 3 auf dem Flop, 4 auf dem Turn, 5 am River
            vis = np.clip(ll[post] + 2, 3, 5)
            vis = np.minimum(vis, nb[hh[post]])
            ok = vis >= 3
            if ok.any():
                sub = np.flatnonzero(post)[ok]
                eq[sub] = equity.postflop_equity(hero[sub], vill[sub],
                                                 board[hh[sub]], vis[ok])
        res.append((idx[w], eq))
    return res
