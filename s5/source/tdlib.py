"""Gemeinsame Bausteine: zensierungsbewusste Typ-Labels und Zaehlregel-Decoder."""
import numpy as np

PB = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_child_samples=20,
          feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
          verbose=-1, deterministic=True, force_col_wise=True)


def build_labels(start, count, erank, ts, typ):
    n = len(erank)
    y1 = np.zeros(n); w1 = np.ones(n); y2 = np.zeros(n); w2 = np.ones(n)
    for a, c in zip(start, count):
        idx = np.arange(a, a + c); li = idx[erank[idx] > 0]; un = idx[erank[idx] == 0]
        k1 = int((typ[li] == 1).sum()); k2 = int((typ[li] == 2).sum())
        t1l = ts[li[typ[li] == 1]].max() if k1 else -1
        t2l = ts[li[typ[li] == 2]].max() if k2 else -1
        y1[li[typ[li] == 1]] = 1; y2[li[typ[li] == 2]] = 1
        if k1 + k2 >= 5:
            if k2 > 0:
                w2[un[ts[un] > t2l]] = 0
            else:
                w1[un[ts[un] > t1l]] = 0
                w2[un] = 0
    return y1, w1, y2, w2


def _prefix(ps, maxk=5):
    dist = np.zeros((len(ps) + 1, maxk + 1)); cur = np.zeros(maxk + 1); cur[0] = 1; dist[0] = cur
    for i, p in enumerate(ps):
        nxt = cur * (1 - p); nxt[1:] += cur[:-1] * p; nxt[maxk] += cur[maxk] * p
        cur = nxt; dist[i + 1] = cur
    return dist


def decode(q1, q2, ts, start, count, cap=5):
    """P(Hand gelistet) = q1*P(<cap Typ1 davor) + q2*P(N1_gesamt + N2_davor < cap)."""
    score = np.zeros(len(q1))
    for a, c in zip(start, count):
        idx = np.arange(a, a + c); o = idx[np.argsort(ts[idx], kind='stable')]
        p1 = np.clip(q1[o], 0, 1); p2 = np.clip(q2[o], 0, 1)
        d1 = _prefix(p1, cap); d2 = _prefix(p2, cap)
        P1 = p1 * d1[:-1, :cap].sum(1)
        tot1 = d1[-1]; cdf2 = np.cumsum(d2[:-1], 1)
        P2 = np.zeros(len(o))
        for aa in range(cap):
            P2 += tot1[aa] * cdf2[:, cap - 1 - aa]
        score[o] = P1 + p2 * P2
    return score
