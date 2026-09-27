"""Equity des Foldenden gegen das TATSAECHLICHE Blatt des Aggressors.

Der schaerfste gemessene Einzelbefund zu directed_transfer: im Moment des Folds
gegen den Partner liegt die Equity des Gebers im Median bei 77,3 % gegen 30,3 %
in gewoehnlichen gemeinsamen Haenden. Unsere bisherigen Merkmale kennen nur
"liegt gerade vorn" (binaer) — Equity ist die abgestufte Groesse und
beruecksichtigt Draws.

Forensisch, nicht unterstellend: der Spieler kennt die Karten des Gegners nicht.
Genau deshalb ist eine systematische Abweichung verdaechtig. Die Karten des
Gegners gehen NUR in die nachtraegliche BEWERTUNG der Entscheidung ein, nie in
die Erwartung an sie (siehe POLICY_FEATURES in engine.py).

Praeflop: 169x169-Klassentabelle, einmalig per Monte Carlo.
Flop:     deterministische Teilmenge der 990 Ausspielungen (feste globale
          Permutation der 1326 Kartenpaare — haengt nur von den Karten ab,
          nicht von IDs oder Zeilenreihenfolge).
Turn:     alle 44 Ausspielungen exakt.
River:    exakter Vergleich, kein Zufall mehr.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np

import engine
import handeval
from engine import A, CACHE

N_PREFLOP_DEALS = 24_000_000
FLOP_SAMPLES = 120
RNG_SEED = 20260907


def hand_class(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """169 Startkartenklassen: suited ueber der Diagonale, offsuit darunter."""
    r1, r2 = c1 // 4, c2 // 4
    s1, s2 = c1 % 4, c2 % 4
    hi, lo = np.maximum(r1, r2), np.minimum(r1, r2)
    suited = s1 == s2
    return np.where(suited, hi * 13 + lo, lo * 13 + hi).astype(np.int16)


def build_preflop_table(n_deals=N_PREFLOP_DEALS, chunk=400_000):
    """EQ[cls_hero, cls_villain] = P(Hero gewinnt) + 0.5 * P(Split)."""
    f = CACHE / "preflop_equity.npy"
    if f.exists():
        return np.load(f)
    rng = np.random.default_rng(RNG_SEED)
    win = np.zeros(169 * 169, dtype=np.float64)
    cnt = np.zeros(169 * 169, dtype=np.float64)
    done = 0
    while done < n_deals:
        m = min(chunk, n_deals - done)
        # m Deals: je 9 verschiedene Karten (2 Hero, 2 Villain, 5 Board)
        d = np.argsort(rng.random((m, 52)), axis=1)[:, :9].astype(np.int16)
        h, v, b = d[:, :2], d[:, 2:4], d[:, 4:9]
        sh = handeval.evaluate(np.concatenate([h, b], axis=1))
        sv = handeval.evaluate(np.concatenate([v, b], axis=1))
        ch = hand_class(h[:, 0], h[:, 1]).astype(np.int64)
        cv = hand_class(v[:, 0], v[:, 1]).astype(np.int64)
        k = ch * 169 + cv
        np.add.at(win, k, (sh > sv) + 0.5 * (sh == sv))
        np.add.at(cnt, k, 1.0)
        done += m
    eq = np.where(cnt > 0, win / np.maximum(cnt, 1), 0.5).reshape(169, 169).astype(np.float32)
    # Symmetrie erzwingen: EQ[a,b] + EQ[b,a] = 1
    eq = 0.5 * (eq + (1.0 - eq.T))
    np.save(f, eq)
    print(f"Praeflop-Equity: {n_deals:,} Deals, "
          f"minimale Zellbelegung {int(cnt.min())}, Median {int(np.median(cnt))}")
    return eq


_COMBOS = None


def _combos():
    """Feste globale Permutation aller 1326 Zweikartenkombinationen."""
    global _COMBOS
    if _COMBOS is None:
        a, b = np.triu_indices(52, 1)
        pairs = np.stack([a, b], axis=1).astype(np.int16)
        perm = np.random.default_rng(RNG_SEED).permutation(len(pairs))
        _COMBOS = pairs[perm]
    return _COMBOS


def postflop_equity(hero, vill, board, nb):
    """hero/vill: (N,2), board: (N,5) mit -1 fuer unbekannt, nb: (N,) 3|4|5."""
    N = len(hero)
    out = np.full(N, np.nan, dtype=np.float32)
    for k in (5, 4, 3):
        m = np.flatnonzero(nb == k)
        if len(m) == 0:
            continue
        h, v, b = hero[m], vill[m], board[m, :k]
        if k == 5:
            sh = handeval.evaluate(np.concatenate([h, b], axis=1))
            sv = handeval.evaluate(np.concatenate([v, b], axis=1))
            out[m] = (sh > sv) + 0.5 * (sh == sv)
            continue
        need = 5 - k
        used = np.concatenate([h, v, b], axis=1)
        cb = _combos()[:, :need] if need == 2 else _combos()[:, :1]
        if need == 1:
            cand = np.arange(52, dtype=np.int16)[None, :]
            ok = ~(cand[..., None] == used[:, None, :]).any(2)     # (n,52)
            sh = np.zeros(len(m)); sv = np.zeros(len(m)); tot = np.zeros(len(m))
            for c in range(52):
                sel = np.flatnonzero(ok[:, c])
                if len(sel) == 0:
                    continue
                bb = np.concatenate([b[sel], np.full((len(sel), 1), c, np.int16)], axis=1)
                a1 = handeval.evaluate(np.concatenate([h[sel], bb], axis=1))
                a2 = handeval.evaluate(np.concatenate([v[sel], bb], axis=1))
                sh[sel] += (a1 > a2); sv[sel] += (a1 < a2); tot[sel] += 1
            out[m] = ((sh + 0.5 * (tot - sh - sv)) / np.maximum(tot, 1)).astype(np.float32)
        else:
            allc = _combos()
            win = np.zeros(len(m)); tie = np.zeros(len(m)); tot = np.zeros(len(m))
            taken = np.zeros(len(m), dtype=np.int32)
            for idx in range(len(allc)):
                c1, c2 = allc[idx]
                live = np.flatnonzero((taken < FLOP_SAMPLES) &
                                      ~((used == c1).any(1) | (used == c2).any(1)))
                if len(live) == 0:
                    if (taken >= FLOP_SAMPLES).all():
                        break
                    continue
                bb = np.concatenate(
                    [b[live], np.full((len(live), 1), c1, np.int16),
                     np.full((len(live), 1), c2, np.int16)], axis=1)
                a1 = handeval.evaluate(np.concatenate([h[live], bb], axis=1))
                a2 = handeval.evaluate(np.concatenate([v[live], bb], axis=1))
                win[live] += (a1 > a2); tie[live] += (a1 == a2)
                tot[live] += 1; taken[live] += 1
            out[m] = ((win + 0.5 * tie) / np.maximum(tot, 1)).astype(np.float32)
    return out
