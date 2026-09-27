"""Directed transfer, role agreement and third-player features for pair hands."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
FEAT = ROOT / "features"
NP_PLAYERS = 12000

NAMES = ["role_flow", "role_flow_agree", "role_flow_mag_agree", "role_net",
         "role_contrib", "role_fold", "role_won", "role_flow_rnet",
         "role_agree_net", "role_colagree", "role_colagree_dir",
         "vict_repeat_share", "vict_pay_max", "vict_pay_sum", "vict_conc",
         "vict_n_payers", "vict_worst_net"]


def _gather(key: np.ndarray, hidx: np.ndarray):
    """Sitzbezogene Rohgroessen je (Paar, Hand) aus dem Cache."""
    p_lo = (key // NP_PLAYERS).astype(np.int32)
    p_hi = (key % NP_PLAYERS).astype(np.int32)
    sp = np.load(CACHE / "seat_player.npy", mmap_mode="r")
    sc = np.load(CACHE / "seat_contrib.npy", mmap_mode="r")
    sn = np.load(CACHE / "seat_net.npy", mmap_mode="r")
    sw = np.load(CACHE / "seat_won_share.npy", mmap_mode="r")
    sf = np.load(CACHE / "seat_folded.npy", mmap_mode="r")
    bb = np.load(CACHE / "hand_bb.npy", mmap_mode="r")
    n = len(key)
    D = {c: np.zeros(n, np.float32) for c in
         ("c_lo", "c_hi", "n_lo", "n_hi", "w_lo", "w_hi", "f_lo", "f_hi",
          "bb", "t3_pay", "t3_pay_max", "t3_n", "t3_net_min")}
    D["vict"] = np.full(n, -1, np.int32)
    CH = 400_000
    for a in range(0, n, CH):
        b = min(a + CH, n)
        h = hidx[a:b]
        spr = np.asarray(sp[h]).astype(np.int32)
        r = np.arange(b - a)
        s_lo = (spr == p_lo[a:b, None]).argmax(1)
        s_hi = (spr == p_hi[a:b, None]).argmax(1)
        C = np.asarray(sc[h]).astype(np.float32)
        N = np.asarray(sn[h]).astype(np.float32)
        W = np.asarray(sw[h]).astype(np.float32)
        F = np.asarray(sf[h]).astype(np.float32)
        bbv = np.maximum(np.asarray(bb[h]).astype(np.float32), 1.0)
        D["bb"][a:b] = bbv
        for nm, M, s in (("c_lo", C, s_lo), ("c_hi", C, s_hi), ("n_lo", N, s_lo),
                         ("n_hi", N, s_hi), ("w_lo", W, s_lo), ("w_hi", W, s_hi),
                         ("f_lo", F, s_lo), ("f_hi", F, s_hi)):
            D[nm][a:b] = M[r, s]
        msk = np.ones_like(C, bool)
        msk[r, s_lo] = False
        msk[r, s_hi] = False
        wpair = (W[r, s_lo] + W[r, s_hi])[:, None]
        pay = np.where(msk, C * wpair, 0.0)
        D["t3_pay"][a:b] = pay.sum(1) / bbv
        j = pay.argmax(1)
        D["t3_pay_max"][a:b] = pay[r, j] / bbv
        D["t3_n"][a:b] = (np.where(msk, C, 0.0) > 0).sum(1)
        D["t3_net_min"][a:b] = np.where(msk, N, np.inf).min(1) / bbv
        D["vict"][a:b] = np.where(pay[r, j] > 0, spr[r, j], -1)
    return D


def _loo_sign(values, key, fallback):
    """sign der Leave-one-out-Summe; Ersatz bei 0 ist selbst antisymmetrisch."""
    s = pd.Series(values).groupby(key).transform("sum").to_numpy()
    r = np.sign(s - values)
    s2 = pd.Series(fallback).groupby(key).transform("sum").to_numpy()
    return np.where(r == 0, np.sign(s2 - fallback), r).astype(np.float32)


def compute(key: np.ndarray, hidx: np.ndarray):
    """(n,17) float32 in der Reihenfolge von `key`/`hidx`, plus Spaltennamen."""
    D = _gather(np.asarray(key), np.asarray(hidx))
    bbv = D["bb"]
    c_lo, c_hi = D["c_lo"] / bbv, D["c_hi"] / bbv
    n_lo, n_hi = D["n_lo"] / bbv, D["n_hi"] / bbv
    w_lo, w_hi, f_lo, f_hi = D["w_lo"], D["w_hi"], D["f_lo"], D["f_hi"]
    sflow = c_lo * w_hi - c_hi * w_lo
    snet, scon, sfo = n_lo - n_hi, c_lo - c_hi, f_lo - f_hi
    r_flow = _loo_sign(sflow, key, -snet)
    r_net = _loo_sign(-snet, key, sflow)
    sgn = np.stack([np.sign(sflow), np.sign(-snet), np.sign(scon), np.sign(sfo)])

    v = D["vict"]
    dd = pd.DataFrame({"key": np.asarray(key), "v": v, "pay": D["t3_pay_max"]})
    tot = dd[dd.v >= 0].groupby(["key", "v"]).pay.sum()
    tot_all = dd[dd.v >= 0].groupby("key").pay.sum()
    tmap = {(int(a), int(b)): float(c) for a, b, c in
            zip(tot.index.get_level_values(0), tot.index.get_level_values(1), tot.to_numpy())}
    amap = tot_all.to_dict()
    own = np.array([tmap.get((int(a), int(b)), 0.0) for a, b in zip(key, v)], np.float32)
    alls = np.array([amap.get(int(a), 0.0) for a in key], np.float32)
    share = np.where(v >= 0, (own - D["t3_pay_max"]) /
                     np.maximum(alls - D["t3_pay_max"], 1e-6), 0.0)

    cols = [sflow * r_flow,
            (sflow * r_flow > 0).astype(np.float32),
            np.abs(sflow) * (sflow * r_flow > 0),
            -snet * r_flow, scon * r_flow, sfo * r_flow, (w_hi - w_lo) * r_flow,
            sflow * r_net, ((-snet) * r_flow > 0).astype(np.float32),
            np.abs(sgn.sum(0)) / 4.0, (sgn.sum(0) * r_flow) / 4.0,
            share, D["t3_pay_max"], D["t3_pay"],
            D["t3_pay_max"] / np.maximum(D["t3_pay"], 1e-6),
            D["t3_n"], D["t3_net_min"]]
    return np.stack(cols, axis=1).astype(np.float32), list(NAMES)


def stored(tag: str = "dev", mask=None):
    """Trainingsseite: Zeilenordnung von `features/ph_key_{tag}.npy`."""
    key = np.load(FEAT / f"ph_key_{tag}.npy")
    hidx = np.load(FEAT / f"ph_hidx_{tag}.npy")
    X, nm = compute(key, hidx)
    return (X if mask is None else X[mask]), nm


def live(key: np.ndarray, hidx: np.ndarray):
    """Vorhersageseite (`predict_evidence._work`): identische Berechnung."""
    return compute(key, hidx)
