"""Eighteen distribution and timing summaries of ordered hand scores."""
import numpy as np

AGG_NAMES = ["inst_max", "inst_top3", "inst_top5", "inst_mean", "inst_sd",
             "inst_share_q90", "inst_share_q99", "inst_n_q99",
             "inst_max_minus_mean", "inst_top3_pos_med", "inst_top3_span",
             "inst_top1_pos",
             # --- B3: Episodenspalten ---
             "inst_gap3", "inst_nbtw3", "inst_gap5", "inst_span5",
             "inst_nbtw5", "inst_top5_pos_med"]

def _agg(uniq, start, cnt, s, tpos, q90, q99):
    """Aggregate je Paar; `s` ist nach (key, torder) sortiert, `tpos` aufsteigend."""
    n = len(uniq)
    A = np.zeros((n, len(AGG_NAMES)), np.float32)
    for i in range(n):
        sl = slice(start[i], start[i] + cnt[i])
        v = s[sl]
        p = tpos[sl]
        o = np.argsort(v)                      # aufsteigend
        k3 = min(3, len(v))
        k5 = min(5, len(v))
        top3 = o[-k3:]
        mu = float(v.mean())
        A[i, 0] = v[o[-1]]
        A[i, 1] = v[top3].mean()
        A[i, 2] = v[o[-k5:]].mean()
        A[i, 3] = mu
        A[i, 4] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        A[i, 5] = float((v > q90).mean())
        A[i, 6] = float((v > q99).mean())
        A[i, 7] = float((v > q99).sum())
        A[i, 8] = A[i, 0] - mu
        pt = p[top3]
        A[i, 9] = float(np.median(pt))
        A[i, 10] = float(pt.max() - pt.min())
        A[i, 11] = float(p[o[-1]])
        # --- B3: Episodenspalten ---------------------------------------
        # Zeilen sind innerhalb des Paares nach torder sortiert -> der
        # Zeilenindex IST die Rangfolge in der Phase.
        i3 = np.sort(top3)
        p3 = p[i3]
        A[i, 12] = float(np.diff(p3).max()) if k3 > 1 else 0.0
        A[i, 13] = float(i3[-1] - i3[0] - (k3 - 1))
        i5 = np.sort(o[-k5:])
        p5 = p[i5]
        A[i, 14] = float(np.diff(p5).max()) if k5 > 1 else 0.0
        A[i, 15] = float(p5[-1] - p5[0])
        A[i, 16] = float(i5[-1] - i5[0] - (k5 - 1))
        A[i, 17] = float(np.median(p5))
    return A
