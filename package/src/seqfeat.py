"""Directed two-step action features for each pair and shared hand."""
from __future__ import annotations

import numpy as np

import engine
import handeval

A_FOLD, A_CHECK, A_CALL, A_BET, A_RAISE, A_ALLIN = 0, 1, 2, 3, 4, 5

DIRECTED = [
    "sq_bet_call",        # i aggressiv -> j callt als naechste eigene Aktion
    "sq_bet_fold",        # i aggressiv -> j foldet
    "sq_bet_raise",       # i aggressiv -> j erhoeht (Konflikt statt Nachgeben)
    "sq_call_amt",        # dieselbe Kette, gewichtet mit Callbetrag/Pot
    "sq_fold_strong",     # j foldet gegen i, obwohl j das bessere Blatt hielt
    "sq_third_fold",      # i aggressiv, ein Dritter foldet, BEVOR j reagiert
    "sq_third_between",   # Zahl der Drittaktionen zwischen i und j
    "sq_alone",           # i aggressiv, kein Dritter mehr lebend
    "sq_yield",           # i verzichtet (check/call) bei hoher eigener Staerke,
                          # waehrend j noch im Pot ist
    "sq_size_pot",        # mittlere Einsatzgroesse von i / Pot, wenn j reagiert
    "sq_size_stack",      # dieselbe Groesse / effektiver Stack
    "sq_potodds",         # Pot-Odds, die j von i gestellt bekommt
]
SCALARS = ["sq_n_events", "sq_n_third_actions", "sq_streets"]
NAMES = ([f"{n}__lh" for n in DIRECTED] + [f"{n}__hl" for n in DIRECTED] + SCALARS)


def table_pair_sequences(hsel):
    A = engine.A
    off, a_seat, a_act = A("act_offsets"), A("act_seat"), A("act_action")
    a_str, a_amt, a_pot = A("act_street"), A("act_amount"), A("act_pot_before")
    a_tc, a_stack = A("act_to_call"), A("act_stack_before")
    sp, bb = A("seat_player"), A("hand_bb").astype(np.float32)
    board, nb_all = A("hand_board"), A("hand_n_board")
    h1, h2 = A("seat_hole1"), A("seat_hole2")
    ii, jj = np.triu_indices(6, 1)
    tord = {int(h): i for i, h in enumerate(hsel)}
    nd = len(DIRECTED)

    keys, hidxs, tords, rows = [], [], [], []
    for h in hsel:
        h = int(h)
        s0, s1 = int(off[h]), int(off[h + 1])
        if s1 - s0 < 2:
            continue
        seats = a_seat[s0:s1].astype(np.int64)
        acts = a_act[s0:s1].astype(np.int64)
        st = a_str[s0:s1].astype(np.int64)
        amt = a_amt[s0:s1].astype(np.float32)
        pot = a_pot[s0:s1].astype(np.float32)
        tc = a_tc[s0:s1].astype(np.float32)
        stk = a_stack[s0:s1].astype(np.float32)
        n = s1 - s0
        aggr = (acts == A_BET) | (acts == A_RAISE) | ((acts == A_ALLIN) & (amt > tc))
        # forensische Endstaerke je Sitz (nur zur Bewertung, nicht als Policy)
        nb = int(nb_all[h])
        holes = np.stack([h1[h], h2[h]], axis=1).astype(np.int16)
        if nb >= 3:
            cards = np.concatenate([holes, np.broadcast_to(board[h, :nb].astype(np.int16),
                                                           (6, nb))], axis=1)
            strength = handeval.evaluate(cards).astype(np.float64)
        else:
            strength = handeval.evaluate(holes).astype(np.float64)
        D = np.zeros((6, 6, nd), dtype=np.float32)
        n_third_actions = 0
        live = np.ones(6, bool)
        for t in range(n):
            if acts[t] == A_FOLD:
                pass
            if not aggr[t]:
                # Verzicht: check/call bei hoher eigener Staerke, Partner lebend
                if acts[t] in (A_CHECK, A_CALL):
                    i = seats[t]
                    for j in range(6):
                        if j != i and live[j] and strength[i] > strength[j]:
                            D[i, j, DIRECTED.index("sq_yield")] += 1.0
                if acts[t] == A_FOLD:
                    live[seats[t]] = False
                continue
            i = seats[t]
            p = max(float(pot[t]), 1e-6)
            size = float(amt[t]) / p
            eff = max(float(stk[t]), 1e-6)
            # naechste eigene Aktion jedes anderen Sitzes in derselben Street
            nxt = {}
            third_before = {}
            cnt_third = 0
            for t2 in range(t + 1, n):
                if st[t2] != st[t]:
                    break
                j = seats[t2]
                if j == i:
                    continue
                if j not in nxt:
                    nxt[j] = t2
                    third_before[j] = cnt_third
                cnt_third += 1
            for j, t2 in nxt.items():
                a2 = acts[t2]
                idx = DIRECTED.index
                if a2 == A_CALL or (a2 == A_ALLIN and amt[t2] <= tc[t2]):
                    D[i, j, idx("sq_bet_call")] += 1.0
                    D[i, j, idx("sq_call_amt")] += float(amt[t2]) / p
                elif a2 == A_FOLD:
                    D[i, j, idx("sq_bet_fold")] += 1.0
                    if strength[j] > strength[i]:
                        D[i, j, idx("sq_fold_strong")] += 1.0
                elif aggr[t2]:
                    D[i, j, idx("sq_bet_raise")] += 1.0
                D[i, j, idx("sq_third_between")] += third_before[j]
                if third_before[j] > 0:
                    for t3 in range(t + 1, t2):
                        if acts[t3] == A_FOLD and seats[t3] != j:
                            D[i, j, idx("sq_third_fold")] += 1.0
                            break
                if live.sum() - 2 <= 0:
                    D[i, j, idx("sq_alone")] += 1.0
                D[i, j, idx("sq_size_pot")] += size
                D[i, j, idx("sq_size_stack")] += float(amt[t]) / eff
                D[i, j, idx("sq_potodds")] += float(tc[t2]) / max(float(pot[t2]) + float(tc[t2]), 1.0)
                n_third_actions += third_before[j]
        pl = sp[h]
        a_id, b_id = pl[ii].astype(np.int64), pl[jj].astype(np.int64)
        lo = np.where(a_id < b_id, ii, jj)
        hi = np.where(a_id < b_id, jj, ii)
        plo, phi = np.minimum(a_id, b_id), np.maximum(a_id, b_id)
        blk = np.concatenate([D[lo, hi], D[hi, lo],
                              np.stack([np.full(15, float((D > 0).sum())),
                                        np.full(15, float(n_third_actions)),
                                        np.full(15, float(st.max() + 1))], axis=1)], axis=1)
        if not np.any(blk[:, :2 * nd]):
            continue
        keys.append(plo * engine.NP_PLAYERS + phi)
        hidxs.append(np.full(15, h, dtype=np.int64))
        tords.append(np.full(15, tord[h], dtype=np.int64))
        rows.append(blk.astype(np.float32))
    if not rows:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros((0, len(NAMES)), np.float32), NAMES)
    key = np.concatenate(keys); hidx = np.concatenate(hidxs)
    torder = np.concatenate(tords); X = np.concatenate(rows, axis=0)
    o = np.lexsort((torder, key))
    return key[o], hidx[o], torder[o], X[o], NAMES
