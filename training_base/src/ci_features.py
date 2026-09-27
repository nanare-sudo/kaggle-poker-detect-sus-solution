"""Partner-fold and multi-player aggression feature for coordinated isolation."""
from __future__ import annotations

import isolation_features as Q3

NAMES = [Q3.NAMES[0]]
assert NAMES == ["q3f_thirdfold_share"], NAMES


def compute(key, hidx):
    X, nm = Q3.compute(key, hidx)
    return X[:, :1].copy(), list(NAMES)


def stored(tag: str = "dev", mask=None):
    X, nm = Q3.stored(tag, mask)
    return X[:, :1].copy(), list(NAMES)


def live(key, hidx):
    return compute(key, hidx)
