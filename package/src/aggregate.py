"""Paar-Aggregation der PH-Merkmale + spielerbereinigte Kontraste.

Koordination ist episodisch (~4 % der gemeinsamen Haende sind Evidenz), deshalb
sind Ordnungsstatistiken (Maximum, Top-k-Mittel, oberes Quantil) und lokal beste
Zeitfenster wichtiger als globale Mittelwerte. Alle Formen werden gebaut, damit
das Modell entscheidet.

Der Kontrast vergleicht das Verhalten gegenueber dem Partner mit dem Verhalten
desselben Spielers gegenueber seinen 28 uebrigen Gegnern im selben Pool
(Leave-one-out). Er ist damit spieler- UND poolbereinigt.
"""
from __future__ import annotations

import numpy as np

import os

TOPK = 3
TOPK2 = 10
QFRACS = tuple(float(x) for x in os.environ.get("AGG_QFRACS", "").split(",") if x.strip())
# AGG_QREPLACE=1 ERSETZT top3/top10 durch die Bruchteilformen, statt sie
# zusaetzlich anzuhaengen. Damit bleibt die Spaltenzahl gleich (kein
# Speicherzuwachs) und der A/B-Vergleich misst genau die Formaenderung.
QREPLACE = bool(int(os.environ.get("AGG_QREPLACE", "0")))
AGG_EPI = bool(int(os.environ.get("AGG_EPI", "0")))
QUANT = 0.90
WINDOWS = (25,)
N_BUCKETS = 6
NP_PLAYERS = 12000


def aggregate_table(key: np.ndarray, torder: np.ndarray, X: np.ndarray, names: list[str]):
    """key/torder/X aus engine.table_pair_hands.

    `torder` ist der chronologische Rang der Hand; danach wird innerhalb jedes
    Paares sortiert, damit die Zeitfenster-Merkmale echte Episoden messen und
    nicht die Dateizeilenreihenfolge.
    """
    order = np.lexsort((torder, key))
    key, X = key[order], X[order]
    uniq, start, cnt = np.unique(key, return_index=True, return_counts=True)
    P, Fn = len(uniq), X.shape[1]

    # Paare eines Pools haben sehr unterschiedlich viele gemeinsame Haende
    # (1..475). Ein gemeinsames Padding auf das Maximum kostet ein Vielfaches,
    # deshalb wird nach Groesse gebucketet. Ergebnis identisch, nur schneller.
    if P > 4 * N_BUCKETS:
        srt = np.argsort(cnt, kind="stable")
        blocks = np.array_split(srt, N_BUCKETS)
        outs = [_agg_block(start[b], cnt[b], X, Fn) for b in blocks]
        inv = np.empty(P, dtype=np.int64)
        inv[np.concatenate(blocks)] = np.arange(P)
        Fm = np.concatenate([o[0] for o in outs], axis=0)[inv]
        sm = np.concatenate([o[1] for o in outs], axis=0)[inv]
        tags = outs[0][2]
    else:
        Fm, sm, tags = _agg_block(start, cnt, X, Fn)
    return _finish(uniq, cnt, Fm, sm, tags, names)


def _agg_block(start, cnt, X, Fn):
    """Ordnungs- und Fensterstatistiken fuer einen Block aehnlich grosser Paare."""
    P = len(start)
    maxn = int(cnt.max())
    idx = np.concatenate([np.arange(s0, s0 + c) for s0, c in zip(start, cnt)])
    Xb = X[idx]
    off = np.concatenate([[0], np.cumsum(cnt[:-1])])
    pos = np.arange(len(Xb)) - np.repeat(off, cnt)
    pad = np.full((P, maxn, Fn), np.nan, dtype=np.float32)
    pad[np.repeat(np.arange(P), cnt), pos] = Xb

    n = cnt.astype(np.float32)[:, None]
    s = np.nansum(pad, axis=1)
    mean = s / n
    with np.errstate(invalid="ignore"):
        var = np.nansum((pad - mean[:, None, :]) ** 2, axis=1) / n
    std = np.sqrt(np.maximum(var, 0.0))

    # EINE Sortierung liefert Maximum, Top-k-Mittel und Quantil (NaN ans Ende)
    sp = np.sort(pad, axis=1)
    ar = np.arange(P)
    mx = sp[ar, cnt - 1]
    topk_mean = np.mean([sp[ar, np.maximum(cnt - 1 - t, 0)] for t in range(TOPK)], axis=0)
    # Top-10-Mittel: gemessen besser als das Quantil, weil die Episoden kurz sind
    kk2 = min(TOPK2, maxn)
    topk2 = np.mean([sp[ar, np.maximum(cnt - 1 - t, 0)] for t in range(kk2)], axis=0)
    sp2 = sp if QFRACS else None
    if not QFRACS:
        del sp

    csum = np.nan_to_num(pad, nan=0.0).cumsum(axis=1)
    wins = []
    for W in WINDOWS:
        if maxn >= W:
            hi = csum[:, W - 1:, :]
            lo = np.concatenate([np.zeros((P, 1, Fn), np.float32), csum[:, : maxn - W, :]], axis=1)
            valid = (np.arange(W - 1, maxn)[None, :] < cnt[:, None])[:, :, None]
            best = np.where(valid, hi - lo, -np.inf).max(axis=1) / W
            best = np.where(np.isfinite(best), best, mean)
        else:
            best = mean
        wins.append(best.astype(np.float32))

    qforms = []
    if QFRACS:
        for fr in QFRACS:
            kk = np.maximum(1, np.rint(fr * cnt).astype(np.int64))
            T = int(kk.max())
            acc = np.zeros((P, Fn), dtype=np.float64)
            for t in range(T):
                v = sp2[ar, np.maximum(cnt - 1 - t, 0)]
                acc += np.where((t < kk)[:, None], np.nan_to_num(v), 0.0)
            qforms.append((acc / kk[:, None]).astype(np.float32))

    epi = []
    if AGG_EPI:
        # (Mittel der drei groessten - Mittel) / Streuung, alles paarintern.
        # Gegen die Expositionsmenge ist die Groesse nicht invariant, aber sie
        # ist gegen die absolute Skala des Paares invariant — genau darum geht es.
        e = (topk_mean - mean) / (std + 1e-6)
        epi = [np.clip(np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 12.0)]

    if QREPLACE and QFRACS:
        blocks = [mean, mx] + qforms + wins + epi
        tags = ["mean", "max"] + [f"q{int(round(fr*1000)):03d}" for fr in QFRACS] \
            + [f"w{W}" for W in WINDOWS] + (["epi3"] if AGG_EPI else [])
    else:
        blocks = [mean, mx, topk_mean, topk2] + wins + qforms + epi
        tags = ["mean", "max", f"top{TOPK}", f"top{TOPK2}"] + [f"w{W}" for W in WINDOWS] \
            + [f"q{int(round(fr*1000)):03d}" for fr in QFRACS] + (["epi3"] if AGG_EPI else [])
    return np.concatenate([b.astype(np.float32) for b in blocks], axis=1), s, tags


def _finish(uniq, cnt, Fm, s, tags, names):
    """Haengt die spielerbereinigten Kontraste an (Leave-one-out im Pool)."""
    Fn = len(names)
    mean, mx = Fm[:, :Fn], Fm[:, Fn:2 * Fn]
    plo, phi = uniq // NP_PLAYERS, uniq % NP_PLAYERS
    players = np.unique(np.concatenate([plo, phi]))
    pmap = {int(p): i for i, p in enumerate(players)}
    ai = np.fromiter((pmap[int(p)] for p in plo), dtype=np.int64, count=len(plo))
    bi = np.fromiter((pmap[int(p)] for p in phi), dtype=np.int64, count=len(phi))
    Ssum = np.zeros((len(players), Fn), dtype=np.float64)
    Ncnt = np.zeros(len(players), dtype=np.float64)
    np.add.at(Ssum, ai, s); np.add.at(Ssum, bi, s)
    np.add.at(Ncnt, ai, cnt); np.add.at(Ncnt, bi, cnt)
    base_a = (Ssum[ai] - s) / np.maximum(Ncnt[ai] - cnt, 1)[:, None]
    base_b = (Ssum[bi] - s) / np.maximum(Ncnt[bi] - cnt, 1)[:, None]
    c_a, c_b = mean - base_a, mean - base_b

    # ---- Lage des Partners unter ALLEN Gegnern des Spielers ---------------
    # Der additive Kontrast (mean - Referenz) misst die absolute Abweichung.
    # Ebenso wichtig ist die RELATIVE Lage: ist dieser Partner der auffaelligste
    # unter den 29 Gegnern des Spielers? Gemessen an directed_transfer gehen
    # 53,6 % des gesamten Chipabflusses eines Gebers an genau einen Partner
    # (harte Negative 13,7 %), und in 70,9 % der Faelle ist der Partner der
    # #1-Gegner (Negative 7,9 %). Diese Form ist aus dem additiven Kontrast
    # nicht rekonstruierbar, weil sie an der Streuung des Spielers normiert.
    NPl = len(players)
    cp = np.zeros(NPl)
    np.add.at(cp, ai, 1.0); np.add.at(cp, bi, 1.0)
    Msum = np.zeros((NPl, Fn)); Msq = np.zeros((NPl, Fn))
    m64 = mean.astype(np.float64)
    np.add.at(Msum, ai, m64); np.add.at(Msum, bi, m64)
    np.add.at(Msq, ai, m64 ** 2); np.add.at(Msq, bi, m64 ** 2)
    zs = []
    for gi in (ai, bi):
        n_loo = np.maximum(cp[gi] - 1, 1)[:, None]
        mu = (Msum[gi] - m64) / n_loo
        var = np.maximum((Msq[gi] - m64 ** 2) / n_loo - mu ** 2, 0.0)
        zs.append((m64 - mu) / np.sqrt(var + 1e-12))
    z_a, z_b = zs
    sh_a = s / (np.abs(Ssum[ai]) + 1e-9)
    sh_b = s / (np.abs(Ssum[bi]) + 1e-9)

    def clip(x):
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)

    # ctr_avg waere exakt (ctr_max + ctr_min)/2 und damit redundant.
    eb = [np.maximum(c_a, c_b), np.minimum(c_a, c_b), mx - 0.5 * (base_a + base_b),
          clip(np.maximum(z_a, z_b)), clip(np.minimum(z_a, z_b)),
          clip(np.maximum(sh_a, sh_b)), clip(np.minimum(sh_a, sh_b))]
    et = ["ctr_max", "ctr_min", "ctrmax", "z_max", "z_min", "share_max", "share_min"]
    F = np.concatenate([Fm] + [b.astype(np.float32) for b in eb], axis=1)
    out_names = [f"{nm}__{tg}" for tg in (tags + et) for nm in names]
    ex = np.stack([cnt.astype(np.float32), np.log1p(cnt.astype(np.float32))], axis=1)
    return uniq, np.concatenate([F, ex], axis=1), out_names + ["n_shared", "log_n_shared"]
