"""Gameplay-aligned action records for evidence ranking.

Preserves the joint situation AT selected decisions, unlike separately pooled
hand statistics. Pair roles are defined by chip outcome, contribution and poker
position, never player IDs. No evidence ordering, generator or label inputs.
"""
from pathlib import Path
import numpy as np
import handeval

CACHE = Path(__file__).resolve().parents[1] / "cache"
_A = {}


def A(name):
    if name not in _A:
        _A[name] = np.load(CACHE / f"{name}.npy", mmap_mode="r")
    return _A[name]


def compute(key, hidx, chunk=1500):
    results = []
    names = None
    for start in range(0, len(key), chunk):
        X, names = _batch(np.asarray(key[start:start+chunk], np.int64),
                          np.asarray(hidx[start:start+chunk], np.int64))
        results.append(X)
    return np.concatenate(results), names


def _batch(key, hs):
    n = len(hs)
    sp = A("seat_player")[hs]
    s0 = (sp == (key // 12000)[:, None]).argmax(1)
    s1 = (sp == (key % 12000)[:, None]).argmax(1)
    assert np.all(sp[np.arange(n), s0] == key // 12000)
    assert np.all(sp[np.arange(n), s1] == key % 12000)
    btn = A("hand_button_seat")[hs].astype(int)
    row = np.arange(n)
    net, con = A("seat_net")[hs], A("seat_contrib")[hs]
    # Lexicographic poker-based orientation. Position is unique on ties.
    v0, v1 = net[row, s0], net[row, s1]
    c0, c1 = con[row, s0], con[row, s1]
    first = (v0 < v1) | ((v0 == v1) & ((c0 > c1) |
            ((c0 == c1) & (((s0-btn) % 6) < ((s1-btn) % 6)))))
    donor, receiver = np.where(first, s0, s1), np.where(first, s1, s0)
    bb = np.maximum(A("hand_bb")[hs].astype(np.float32), 1)[:, None]
    off = A("act_offsets")
    ln = off[hs+1] - off[hs]
    length = int(ln.max())
    mask = np.arange(length)[None] < ln[:, None]
    ix = np.minimum(off[hs, None] + np.arange(length), off[hs+1, None]-1)
    get = lambda name: np.asarray(A(name)[ix])
    seat = get("act_seat").astype(int)
    act, st = get("act_action"), get("act_street").astype(int)
    amt, tc, pot, stack = [get(x).astype(np.float32) for x in
        ("act_amount", "act_to_call", "act_pot_before", "act_stack_before")]
    aggr = ((act == 3) | (act == 4) | ((act == 5) & (amt > tc))) & mask
    role = np.where(seat == donor[:, None], 0, np.where(seat == receiver[:, None], 1, 2))
    prev_role = np.full((n, length), 3, np.float32)
    prev_partner_act = np.full((n, length), 6, np.float32)
    alive = np.ones((n, 6), bool)
    palive = np.zeros((n, length), np.float32)
    nbetter = np.zeros((n, length), np.float32)
    third_best = np.zeros((n, length), np.float32)
    third_alive = np.zeros((n, length), np.float32)
    prev = np.full(n, 3, np.float32)
    last_act = np.full((n, 6), 6, np.float32)
    holes = np.stack([A("seat_hole1")[hs], A("seat_hole2")[hs]], 2)
    board = A("hand_board")[hs]
    strength = np.empty((n, 4, 6), np.float64)
    for street, nc in enumerate((0, 3, 4, 5)):
        cards = holes if nc == 0 else np.concatenate([
            holes, np.broadcast_to(board[:, None, :nc], (n, 6, nc))], 2)
        strength[:, street] = handeval.evaluate(cards.reshape(-1, cards.shape[-1])).reshape(n, 6)
    sraw = strength[row[:, None], st, seat]
    cat = np.floor(sraw / handeval.B**6)
    partner = np.where(role == 0, receiver[:, None], donor[:, None])
    praw = strength[row[:, None], st, partner]
    pcat = np.floor(praw / handeval.B**6)
    for t in range(length):
        if t:
            prev = np.where(st[:, t] != st[:, t-1], 3, prev)
        prev_role[:, t] = prev
        prev_partner_act[:, t] = last_act[row, partner[:, t]]
        palive[:, t] = alive[row, partner[:, t]]
        cur = strength[row, st[:, t]]
        nbetter[:, t] = ((cur > sraw[:, t, None]) & alive).sum(1)
        other = alive & (np.arange(6)[None] != donor[:, None]) & (np.arange(6)[None] != receiver[:, None])
        third_best[:, t] = np.where(other, cur / handeval.B**6, -1).max(1)
        third_alive[:, t] = other.sum(1)
        prev = np.where(aggr[:, t], role[:, t], prev)
        last_act[row, seat[:, t]] = np.where(mask[:, t], act[:, t], last_act[row, seat[:, t]])
        alive[row, seat[:, t]] &= ~((act[:, t] == 0) & mask[:, t])
    sur = get("act_surprise_q2nnnoply")
    eq = get("act_equity")
    fields = [st, act, (seat-btn[:, None]) % 6, (partner-seat) % 6,
              get("act_players_active"), palive, prev_role, prev_partner_act,
              cat, pcat, (sraw-praw) / handeval.B**6, nbetter, third_best, third_alive,
              np.log1p(amt/bb), amt/np.maximum(pot, 1), tc/np.maximum(pot+tc, 1),
              np.log1p(pot/bb), np.log1p(stack/bb), tc/np.maximum(stack, 1),
              sur, get("act_paggr_q2nnnoply"), get("act_pfold_q2nnnoply"),
              np.nan_to_num(eq, nan=0.5), np.isfinite(eq).astype(float)]
    fn = ["street", "action", "position", "partner_position", "active", "partner_alive",
          "last_aggressor_role", "partner_previous_action", "own_category", "partner_category",
          "own_minus_partner_strength", "better_live", "third_best", "third_alive", "log_amount_bb",
          "amount_pot", "pot_odds", "log_pot_bb", "log_stack_bb", "call_stack",
          "surprise", "p_aggr", "p_fold", "equity", "equity_known"]
    tensor = np.stack(fields, 2).astype(np.float32)
    blocks, names = [], []
    for rr, rname in ((0, "loser"), (1, "winner")):
        own = mask & (role == rr)
        selections = [
            ("fold", own & (act == 0), np.broadcast_to(np.arange(length), mask.shape)),
            ("passive", own & np.isin(act, [1, 2, 5]) & ~aggr, sur),
            ("aggr", own & aggr, amt/np.maximum(pot, 1)),
            ("surprising", own, sur),
        ]
        for event, eligible, priority in selections:
            idx = np.where(eligible, priority, -np.inf).argmax(1)
            exists = eligible.any(1)
            rec = tensor[row, idx].copy()
            rec[~exists] = 0
            blocks.extend([exists[:, None].astype(np.float32), rec])
            prefix = f"evt_{rname}_{event}_"
            names.extend([prefix+"exists"] + [prefix+f for f in fn])
    return np.concatenate(blocks, 1), names
