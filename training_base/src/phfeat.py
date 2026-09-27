"""Anreicherung der PH-Merkmale um paarinterne Normalisierung.

Der Evidenz-Ranker sortiert HAENDE INNERHALB eines Paares. Die absolute Hoehe
eines Merkmals ist dafuer weniger aussagekraeftig als seine Lage relativ zu dem,
was dieses Paar sonst tut. Deshalb je Rohmerkmal zusaetzlich:
  z   — (x - Paarmittel) / Paar-Standardabweichung
  rnk — Rang innerhalb des Paares, auf 0..1 normiert
"""
from __future__ import annotations

import numpy as np


def enrich(key: np.ndarray, X: np.ndarray, names: list[str], torder=None,
           rank_ties="ordinal", gap_fill="global"):
    """Add within-pair z-scores, ranks and chronology. Keys must be grouped.
    Time features and chronological tie-breaks use explicit hand timestamps.
    """
    if rank_ties not in ("ordinal", "average", "chronological") or gap_fill not in ("global", "pair"):
        raise ValueError("unsupported rank_ties or gap_fill")
    if rank_ties == "chronological" and torder is None:
        raise ValueError("chronological ranks require explicit gameplay time order")
    uniq, start, cnt = np.unique(key, return_index=True, return_counts=True)
    gid = np.repeat(np.arange(len(uniq)), cnt)
    n = cnt.astype(np.float32)[:, None]
    s = np.zeros((len(uniq), X.shape[1]), dtype=np.float64)
    np.add.at(s, gid, X)
    mean = (s / n).astype(np.float32)
    d = X - mean[gid]
    s2 = np.zeros_like(s)
    np.add.at(s2, gid, d.astype(np.float64) ** 2)
    sd = np.sqrt(s2 / n).astype(np.float32)
    z = d / np.maximum(sd[gid], 1e-6)

    # Rang innerhalb des Paares (0 = kleinster, 1 = groesster)
    rnk = np.empty_like(X)
    ar = np.arange(len(X))
    for f in range(X.shape[1]):
        o = (np.lexsort((np.asarray(torder), X[:, f], key))
             if rank_ties == "chronological" else np.lexsort((X[:, f], key)))
        r = np.empty(len(X), dtype=np.float32)
        sorted_ranks = ar - np.repeat(start, cnt)
        if rank_ties == "average":
            # Equal observations carry equal rank, independently of input order.
            changes = np.r_[True, (key[o][1:] != key[o][:-1]) |
                                  (X[o, f][1:] != X[o, f][:-1])]
            a = np.flatnonzero(changes)
            b = np.r_[a[1:], len(o)]
            sorted_ranks = np.repeat((sorted_ranks[a] + sorted_ranks[b-1])/2, b-a)
        r[o] = sorted_ranks
        rnk[:, f] = r / np.maximum(np.repeat(cnt - 1, cnt), 1)
    blocks, ne = [X, z, rnk], names + [f"{m}__z" for m in names] + [f"{m}__rnk" for m in names]
    if torder is not None:
        t = np.asarray(torder, dtype=np.float64)
        n1 = np.maximum(np.repeat(cnt - 1, cnt), 1)

        # chronologische Ordnung INNERHALB des Paares (nicht die Zeilenordnung)
        o = np.lexsort((t, key))
        pos = np.empty(len(X), dtype=np.int64)
        pos[o] = ar - np.repeat(start, cnt)      # 0 = zeitlich erste Hand des Paares

        ts = t[o]
        prev_s = np.empty(len(X)); prev_s[:] = np.nan
        nxt_s = np.empty(len(X)); nxt_s[:] = np.nan
        prev_s[1:] = ts[1:] - ts[:-1]; nxt_s[:-1] = ts[1:] - ts[:-1]
        pos_s = ar - np.repeat(start, cnt)
        prev_s[pos_s == 0] = np.nan
        nxt_s[pos_s == np.repeat(cnt - 1, cnt)] = np.nan
        prev = np.empty(len(X)); prev[o] = prev_s
        nxt = np.empty(len(X)); nxt[o] = nxt_s
        med = np.nanmedian(prev) if np.isfinite(prev).any() else 1.0
        if gap_fill == "pair":
            # Serving batches must not alter edge values for another pair.
            pair_med = np.array([np.median(prev[a:a+c][np.isfinite(prev[a:a+c])])
                                 if np.isfinite(prev[a:a+c]).any() else 1.0
                                 for a,c in zip(start,cnt)])
            replacement = np.repeat(pair_med,cnt)
            prev = np.where(np.isfinite(prev),prev,replacement)
            nxt = np.where(np.isfinite(nxt),nxt,replacement)

        tmin_row = np.empty(len(X)); span_row = np.empty(len(X))
        tmin_row[o] = np.repeat(np.minimum.reduceat(ts, start), cnt)
        span_row[o] = np.maximum(
            np.repeat(np.maximum.reduceat(ts, start)
                      - np.minimum.reduceat(ts, start), cnt), 1)
        extra = np.stack([
            pos / n1,                                   # relative Lage 0..1
            np.log1p(pos),                              # absolute Lage
            np.log1p(np.nan_to_num(prev, nan=med)),     # Abstand zur Vorhand
            np.log1p(np.nan_to_num(nxt, nan=med)),      # Abstand zur Folgehand
            (t - tmin_row) / span_row,                  # Zeitanteil
        ], axis=1).astype(np.float32)
        blocks.append(extra)
        ne = ne + ["chrono_relpos", "chrono_logpos", "chrono_gap_prev",
                   "chrono_gap_next", "chrono_timefrac"]
    Xe = np.concatenate(blocks, axis=1)
    return Xe.astype(np.float32), ne
