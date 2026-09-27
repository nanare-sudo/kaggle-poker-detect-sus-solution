"""Stored and live interfaces to directed action-sequence features."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FEAT = Path(__file__).resolve().parent.parent / "features"


def _ndir(sn):
    """Read the number of directional features from the schema."""
    n = sum(1 for x in sn if x.endswith("__lh"))
    assert sum(1 for x in sn if x.endswith("__hl")) == n, sn
    return n


def enabled() -> bool:
    return os.environ.get("SEQ_FEAT", "") == "1"


def _names():
    sn = json.loads((FEAT / "sq_ph_names.json").read_text())
    nd = _ndir(sn)
    base = [n.replace("__lh", "") for n in sn[:nd]]
    return ([f"{n}_sum" for n in base] + [f"{n}_absdiff" for n in base]
            + [f"{n}_max" for n in base] + [f"{n}_min" for n in base] + sn[2 * nd:])


def block(S: np.ndarray):
    """Rohmatrix (n, 2*nd+s) -> symmetrisierter Block plus Spaltennamen."""
    nd = _ndir(json.loads((FEAT / "sq_ph_names.json").read_text()))
    assert S.shape[1] >= 2 * nd, (S.shape, nd)
    lh, hl, sc = S[:, :nd], S[:, nd:2 * nd], S[:, 2 * nd:]
    B = np.concatenate([lh + hl, np.abs(lh - hl), np.maximum(lh, hl),
                        np.minimum(lh, hl), sc], axis=1)
    return B.astype(np.float32), _names()


def stored(tag: str, mask=None):
    """Trainingsseite: die von seqbuild2 abgelegte Matrix."""
    S = np.load(FEAT / f"sq_ph_{tag}.npy")
    n_ph = len(np.load(FEAT / f"ph_key_{tag}.npy", mmap_mode="r"))
    assert len(S) == n_ph, (
        f"sq_ph_{tag} hat {len(S)} Zeilen, ph_key_{tag} aber {n_ph}. "
        "Erst `python3 src/seqbuild2.py " + tag + "` neu laufen lassen — ein "
        "stilles Verrutschen der Zuordnung waere sonst nicht zu bemerken.")
    if mask is not None:
        S = S[mask]
    return block(S)


def live(hsel, key, hidx):
    """Vorhersageseite: dieselben Groessen live aus dem Motor, auf (key, hidx)
    der PH-Zeilen ausgerichtet. Nicht getroffene Zeilen bleiben null — genau wie
    im Training, wo 93,5 % der Zeilen belegt waren."""
    import seqfeat
    sk, sh, _, SX, _ = seqfeat.table_pair_sequences(hsel)
    out = np.zeros((len(key), SX.shape[1]), np.float32)
    if len(sk):
        src = sk.astype(np.int64) * (10 ** 7) + sh.astype(np.int64)
        dst = key.astype(np.int64) * (10 ** 7) + hidx.astype(np.int64)
        o = np.argsort(src)
        pos = np.searchsorted(src[o], dst)
        hit = (pos < len(src)) & (src[o][np.minimum(pos, len(src) - 1)] == dst)
        out[hit] = SX[o][pos[hit]]
    return block(out)
