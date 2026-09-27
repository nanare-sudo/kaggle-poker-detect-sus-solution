"""Vektorisierte Pokerhand-Bewertung fuer 2 bis 7 Karten (reine Tabellenzugriffe).

Liefert je Hand eine vergleichbare int64-Staerke (groesser = besser). Wird nur
FORENSISCH benutzt (nachtraegliche Auswertung "wer haette gewonnen"), nie um
einem Spieler Wissen ueber zukuenftige Boardkarten zu unterstellen: fuer
Entscheidungen zum Aktionszeitpunkt werden ausschliesslich die dann sichtbaren
Karten uebergeben.

Kartencode: card = rank*4 + suit, rank 0..12 (2..A), suit 0..3. -1 = keine Karte.
Verifiziert gegen Brute-Force best-5-aus-7 (100 % Uebereinstimmung, 4000 Haende).
"""
from __future__ import annotations

import numpy as np

CAT_HIGH, CAT_PAIR, CAT_TWOPAIR, CAT_TRIPS = 0, 1, 2, 3
CAT_STRAIGHT, CAT_FLUSH, CAT_FULL, CAT_QUADS, CAT_SF = 4, 5, 6, 7, 8
B = 13
NM = 1 << 13


def _tables():
    straight = np.full(NM, -1, dtype=np.int64)
    msb = np.full(NM, -1, dtype=np.int64)
    topk = np.zeros((6, NM), dtype=np.int64)   # topk[k] = Kodierung der k hoechsten Raenge
    pw = [B ** 4, B ** 3, B ** 2, B, 1]
    for m in range(NM):
        rs = [r for r in range(12, -1, -1) if m & (1 << r)]
        if rs:
            msb[m] = rs[0]
        best = -1
        for hi in range(12, 3, -1):
            if all(m & (1 << (hi - d)) for d in range(5)):
                best = hi
                break
        if best < 0 and (m & (1 << 12)) and (m & 0b1111) == 0b1111:
            best = 3           # Wheel A-2-3-4-5
        straight[m] = best
        for k in range(6):
            v = 0
            for i, r in enumerate(rs[:k]):
                v += (r + 1) * pw[i]
            topk[k, m] = v
    return straight, msb, topk


STRAIGHT_HIGH, MSB, TOPK = _tables()
_BITS = (1 << np.arange(13)).astype(np.int64)
_P = np.array([B ** 4, B ** 3, B ** 2, B, 1], dtype=np.int64)


def evaluate(cards: np.ndarray) -> np.ndarray:
    """cards: (N, C) Kartencodes 0..51, -1 = keine Karte. -> (N,) int64 Staerke.

    Nur innerhalb derselben Hand vergleichen (gleiche Kartenanzahl / gleiches Board).
    """
    cards = np.asarray(cards)
    N, C = cards.shape
    valid = cards >= 0
    rank = np.where(valid, cards // 4, 0).astype(np.int64)
    suit = np.where(valid, cards % 4, 0).astype(np.int64)
    rows = np.repeat(np.arange(N), C)

    rc = np.zeros((N, 13), dtype=np.int8)
    np.add.at(rc, (rows, rank.ravel()), valid.ravel().astype(np.int8))
    sc = np.zeros((N, 4), dtype=np.int8)
    np.add.at(sc, (rows, suit.ravel()), valid.ravel().astype(np.int8))

    rmask = ((rc > 0).astype(np.int64) * _BITS).sum(1)
    fs = sc.argmax(1)
    has_flush = sc.max(1) >= 5
    fmask = np.zeros(N, dtype=np.int64)
    sel = (valid & (suit == fs[:, None])).ravel()
    np.bitwise_or.at(fmask, rows[sel], _BITS[rank.ravel()[sel]])

    m4 = ((rc == 4).astype(np.int64) * _BITS).sum(1)
    m3 = ((rc == 3).astype(np.int64) * _BITS).sum(1)
    m2 = ((rc == 2).astype(np.int64) * _BITS).sum(1)
    n3, n2 = (rc == 3).sum(1), (rc == 2).sum(1)
    r4, r3, r2 = MSB[m4], MSB[m3], MSB[m2]
    r3b = MSB[np.where(r3 >= 0, m3 & ~_BITS[np.maximum(r3, 0)], m3)]
    r2b = MSB[np.where(r2 >= 0, m2 & ~_BITS[np.maximum(r2, 0)], m2)]

    cat = np.full(N, CAT_HIGH, dtype=np.int64)
    tb = TOPK[5, rmask]

    # ein Paar
    m = n2 >= 1
    k1 = np.where(r2 >= 0, rmask & ~_BITS[np.maximum(r2, 0)], rmask)
    cat = np.where(m, CAT_PAIR, cat)
    tb = np.where(m, (r2 + 1) * _P[0] + TOPK[3, k1] // B, tb)
    # zwei Paare
    m = n2 >= 2
    k2 = np.where(r2b >= 0, k1 & ~_BITS[np.maximum(r2b, 0)], k1)
    cat = np.where(m, CAT_TWOPAIR, cat)
    tb = np.where(m, (r2 + 1) * _P[0] + (r2b + 1) * _P[1] + TOPK[1, k2] // (B ** 2), tb)
    # Drilling (nur wenn kein Full House daraus wird)
    m = (n3 == 1) & (n2 == 0)
    k3 = np.where(r3 >= 0, rmask & ~_BITS[np.maximum(r3, 0)], rmask)
    cat = np.where(m, CAT_TRIPS, cat)
    tb = np.where(m, (r3 + 1) * _P[0] + TOPK[2, k3] // B, tb)
    # Strasse
    m = STRAIGHT_HIGH[rmask] >= 0
    cat = np.where(m, CAT_STRAIGHT, cat)
    tb = np.where(m, (STRAIGHT_HIGH[rmask] + 1) * _P[0], tb)
    # Flush
    cat = np.where(has_flush, CAT_FLUSH, cat)
    tb = np.where(has_flush, TOPK[5, fmask], tb)
    # Full House
    m = (n3 >= 1) & ((n2 >= 1) | (n3 >= 2))
    pf = np.where(n3 >= 2, np.maximum(r3b, r2), r2)
    cat = np.where(m, CAT_FULL, cat)
    tb = np.where(m, (r3 + 1) * _P[0] + (pf + 1) * _P[1], tb)
    # Vierling
    k4 = np.where(r4 >= 0, rmask & ~_BITS[np.maximum(r4, 0)], rmask)
    cat = np.where(m4 > 0, CAT_QUADS, cat)
    tb = np.where(m4 > 0, (r4 + 1) * _P[0] + TOPK[1, k4] // B, tb)
    # Straight Flush
    m = has_flush & (STRAIGHT_HIGH[fmask] >= 0)
    cat = np.where(m, CAT_SF, cat)
    tb = np.where(m, (STRAIGHT_HIGH[fmask] + 1) * _P[0], tb)

    return cat * (B ** 6) + tb
