"""Action context and player statistics with phase-specific, leave-hand-out estimates."""
from __future__ import annotations

import os

os.environ.setdefault("POLICY_FIX", "1")
os.environ.setdefault("POLICY_V2", "1")

from pathlib import Path

import numpy as np

import engine
from engine import A, A_FOLD, A_CHECK, A_CALL

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
STATS = Path(os.environ.get("Q2_STATS_FILE", str(OUT / "q2_pstats.npz")))
# Q2_POOL=1: Gewohnheiten je Spieler ueber BEIDE Phasen gepoolt (Stil ist phasenstabil, 2026-09-13).
Q2_POOL = bool(int(os.environ.get("Q2_POOL", "0")))
# Q2_WIN_HEAD=N: Zaehler der dev-Phase nur aus den ersten N Haenden je Tisch (Auswertungsexposition nachstellen).
Q2_WIN_HEAD = int(os.environ.get("Q2_WIN_HEAD", "0"))
NPL = 12000

# ---------------------------------------------------------------------------
# Zaehlerdefinition
# ---------------------------------------------------------------------------
ACT_CTR = [
    "n_acts", "n_aggr", "n_fold", "n_call", "n_check",
    "acts_s0", "acts_s1", "acts_s2", "acts_s3",
    "aggr_s0", "aggr_s1", "aggr_s2", "aggr_s3",
    "faced", "fold_faced", "call_faced", "aggr_faced",
    "free", "bet_free", "sz_sum", "tb_opp", "tb_act",
]
HND_CTR = ["hands", "hands_pre", "hands_vpip", "hands_pfr", "hands_nofold"]
KA, KH = len(ACT_CTR), len(HND_CTR)

# (Name, Zaehlerindex in ACT/HND, Nennerindex, Quelle)  Quelle: "a"=Aktion, "h"=Hand
RATES = [
    ("q2p_vpip",      "h", 2, 1), ("q2p_pfr",       "h", 3, 1),
    ("q2p_3bet",      "a", 21, 20),
    ("q2p_af_pre",    "a", 9, 5), ("q2p_af_flop",   "a", 10, 6),
    ("q2p_af_turn",   "a", 11, 7), ("q2p_af_river", "a", 12, 8),
    ("q2p_foldto",    "a", 14, 13), ("q2p_callto",  "a", 15, 13),
    ("q2p_raiseto",   "a", 16, 13), ("q2p_betfree", "a", 18, 17),
    ("q2p_wtsd",      "h", 4, 0),
    ("q2p_betsize",   "a", 19, 1),
    ("q2p_paggr",     "a", 1, 0), ("q2p_pfold",     "a", 2, 0),
    ("q2p_pcall",     "a", 3, 0), ("q2p_pcheck",    "a", 4, 0),
]
PLAYER_COLS = [r[0] for r in RATES] + ["q2p_acts_per_hand", "q2p_n_hands"]
D_COLS = ["q2_n_to_act", "q2_n_behind", "q2_is_pfa"]
SMOOTH = 25.0          # Empirische-Bayes-Staerke in "Beobachtungen"

BLOCKS = {
    "base13": list(range(13)),
    "v2": list(range(13, 22)),
    "d3": list(range(22, 25)),
    "player": list(range(25, 25 + len(PLAYER_COLS))),
}
N_COLS = 25 + len(PLAYER_COLS)


def col_names():
    return (list(engine.POLICY_FEATURES) + D_COLS + PLAYER_COLS)


# ---------------------------------------------------------------------------
# 1. globale Gewohnheitszaehler je (Spieler, Phase)
# ---------------------------------------------------------------------------
def _act_contrib(street, act, to_call, pot, amount, bb_h):
    """(n, KA)-Beitragsmatrix je Aktion (float32)."""
    n = len(act)
    is_aggr = (act == engine.A_BET) | (act == engine.A_RAISE) | (
        (act == engine.A_ALLIN) & (amount > to_call))
    is_fold = act == A_FOLD
    is_call = (act == A_CALL) | ((act == engine.A_ALLIN) & ~is_aggr)
    is_check = act == A_CHECK
    faced = to_call > 0
    free = ~faced
    C = np.zeros((n, KA), np.float32)
    C[:, 0] = 1.0
    C[:, 1] = is_aggr
    C[:, 2] = is_fold
    C[:, 3] = is_call
    C[:, 4] = is_check
    for s in range(4):
        m = street == s
        C[:, 5 + s] = m
        C[:, 9 + s] = m & is_aggr
    C[:, 13] = faced
    C[:, 14] = faced & is_fold
    C[:, 15] = faced & is_call
    C[:, 16] = faced & is_aggr
    C[:, 17] = free
    C[:, 18] = free & is_aggr
    C[:, 19] = np.where(is_aggr, amount / np.maximum(pot + to_call, 1.0), 0.0)
    opp = (street == 0) & (to_call > bb_h)          # praeflop gegen eine Erhoehung
    C[:, 20] = opp
    C[:, 21] = opp & is_aggr
    return C


def build_stats(verbose=True):
    """Zaehlt alle 18,6 Mio Aktionen einmal durch und legt die Totale ab."""
    hidx = np.asarray(A("act_hidx")).astype(np.int64)
    seat = np.asarray(A("act_seat")).astype(np.int64)
    act = np.asarray(A("act_action"))
    street = np.asarray(A("act_street")).astype(np.int64)
    to_call = np.asarray(A("act_to_call")).astype(np.float32)
    pot = np.asarray(A("act_pot_before")).astype(np.float32)
    amount = np.asarray(A("act_amount")).astype(np.float32)
    hp = np.asarray(A("hand_phase")).astype(np.int64)
    bb = np.asarray(A("hand_bb")).astype(np.float32)
    sp = np.asarray(A("seat_player")).astype(np.int64)
    folded = np.asarray(A("seat_folded"))

    player = sp[hidx, seat]
    phase = hp[hidx]
    gid = player * 2 + phase
    if Q2_WIN_HEAD:
        import hand_order as _CH
        _rk = _CH._hand_order()
        keep_a = (phase == 1) | ((_rk[hidx] >= 0) & (_rk[hidx] < Q2_WIN_HEAD))
        gid = np.where(keep_a, gid, NPL * 2)          # Ueberlaufzelle, wird verworfen
    ACT_TOT = np.zeros((NPL * 2, KA), np.float64)
    CH = 2_000_000                       # gechunkt: (18,6 Mio x 22) float32 waeren 1,6 GB
    for a0 in range(0, len(act), CH):
        b0 = min(a0 + CH, len(act))
        C = _act_contrib(street[a0:b0], act[a0:b0], to_call[a0:b0], pot[a0:b0],
                         amount[a0:b0], bb[hidx[a0:b0]])
        g = gid[a0:b0]
        for k in range(KA):
            ACT_TOT[:, k] += np.bincount(g, weights=C[:, k].astype(np.float64),
                                         minlength=NPL * 2 + 1)[:NPL * 2]
        del C

    # Handebene: jeder Sitz jeder Hand ist eine Beobachtung
    H = len(hp)
    hseat_p = sp.reshape(-1)                                   # (H*6,)
    hseat_ph = np.repeat(phase_of := hp, 6)
    ghs = hseat_p * 2 + hseat_ph
    if Q2_WIN_HEAD:
        keep_h = np.repeat((hp == 1) | ((_rk >= 0) & (_rk < Q2_WIN_HEAD)), 6)
        ghs = np.where(keep_h, ghs, NPL * 2)
    HND_TOT = np.zeros((NPL * 2, KH), np.float64)
    HND_TOT[:, 0] = np.bincount(ghs, minlength=NPL * 2 + 1)[:NPL * 2]
    HND_TOT[:, 4] = np.bincount(ghs, weights=(~folded).reshape(-1).astype(np.float64),
                                minlength=NPL * 2 + 1)[:NPL * 2]
    # hands_pre / vpip / pfr: ueber die Aktionen, je (Hand, Sitz) nur einmal
    is_aggr = (act == engine.A_BET) | (act == engine.A_RAISE) | (
        (act == engine.A_ALLIN) & (amount > to_call))
    is_vol = (act == A_CALL) | is_aggr                     # freiwillig Geld in den Pot
    pre = street == 0
    hs_key = hidx * 6 + seat
    for col, mask in ((1, pre), (2, pre & is_vol), (3, pre & is_aggr)):
        k = np.unique(hs_key[mask])
        g = sp.reshape(-1)[k] * 2 + hseat_ph[k]
        if Q2_WIN_HEAD:
            g = np.where(keep_h[k], g, NPL * 2)
        HND_TOT[:, col] = np.bincount(g, minlength=NPL * 2 + 1)[:NPL * 2]
    del phase_of
    np.savez(STATS, ACT_TOT=ACT_TOT.astype(np.float64),
             HND_TOT=HND_TOT.astype(np.float64),
             act_ctr=np.array(ACT_CTR), hnd_ctr=np.array(HND_CTR))
    if verbose:
        a = ACT_TOT.sum(0)
        print(f"Aktionen gesamt {a[0]:,.0f}  aggr {a[1]/a[0]:.4f} fold {a[2]/a[0]:.4f} "
              f"call {a[3]/a[0]:.4f} check {a[4]/a[0]:.4f}", flush=True)
        h = HND_TOT.sum(0)
        print(f"Sitzhaende {h[0]:,.0f}  VPIP {h[2]/h[1]:.4f}  PFR {h[3]/h[1]:.4f}  "
              f"nofold {h[4]/h[0]:.4f}", flush=True)
        nz = HND_TOT[:, 0] > 0
        print(f"belegte (Spieler,Phase)-Zellen {nz.sum():,} von {NPL*2:,}; "
              f"Haende je Zelle Median {np.median(HND_TOT[nz,0]):.0f}", flush=True)
    return ACT_TOT, HND_TOT


_ST = None


def load_stats():
    global _ST
    if _ST is None:
        if not STATS.exists():
            build_stats()
        z = np.load(STATS, allow_pickle=True)
        ACT_TOT, HND_TOT = z["ACT_TOT"], z["HND_TOT"]
        if Q2_POOL:
            ACT_TOT = ACT_TOT.copy(); HND_TOT = HND_TOT.copy()
            pa = ACT_TOT[0::2] + ACT_TOT[1::2]; ph_ = HND_TOT[0::2] + HND_TOT[1::2]
            ACT_TOT[0::2] = pa; ACT_TOT[1::2] = pa; HND_TOT[0::2] = ph_; HND_TOT[1::2] = ph_
        pop = np.zeros(len(RATES), np.float64)
        for i, (_, src, num, den) in enumerate(RATES):
            T = ACT_TOT if src == "a" else HND_TOT
            pop[i] = T[:, num].sum() / max(T[:, den].sum(), 1.0)
        pop_aph = ACT_TOT[:, 0].sum() / max(HND_TOT[:, 0].sum(), 1.0)
        _ST = (ACT_TOT, HND_TOT, pop, pop_aph)
    return _ST


def _rates(act_num, hnd_num):
    """Raten aus (m, KA)/(m, KH)-Zaehlern; Glaettung gegen die Population."""
    _, _, pop, pop_aph = load_stats()
    out = np.empty((len(act_num), len(PLAYER_COLS)), np.float32)
    for i, (_, src, num, den) in enumerate(RATES):
        T = act_num if src == "a" else hnd_num
        out[:, i] = (T[:, num] + SMOOTH * pop[i]) / (T[:, den] + SMOOTH)
    out[:, len(RATES)] = (act_num[:, 0] + SMOOTH * pop_aph) / (hnd_num[:, 0] + SMOOTH)
    out[:, len(RATES) + 1] = hnd_num[:, 0]
    return out


# ---------------------------------------------------------------------------
# 2. Block 2: Zugreihenfolge und Praeflop-Aggressor (nur VORHERIGE Aktionen)
# ---------------------------------------------------------------------------
def order_block(hloc, a_str, a_seat, a_act, btn_h, is_aggr):
    """n_to_act, n_behind, is_pfa je Aktion.

    n_to_act  Zahl der lebenden Gegner, die nach mir in DIESER Setzrunde noch
              handeln muessen (seit der letzten Erhoehung nicht gehandelt).
    n_behind  Zahl der lebenden Gegner, die in der Reihenfolge dieser Street
              hinter mir sitzen (unabhaengig von der Orbitlage).
    is_pfa    bin ich (bisher) der Praeflop-Aggressor.
    """
    n = len(hloc)
    pos = np.arange(n, dtype=np.int64)
    grp = hloc * 4 + a_str
    new_hs = np.empty(n, bool)
    new_hs[0] = True
    new_hs[1:] = grp[1:] != grp[:-1]
    new_h = np.empty(n, bool)
    new_h[0] = True
    new_h[1:] = hloc[1:] != hloc[:-1]

    start_hs = np.maximum.accumulate(np.where(new_hs, pos, 0))
    start_h = np.maximum.accumulate(np.where(new_h, pos, 0))

    def cummax_prev(vals, start):
        """Groesste Trefferposition j < i in derselben Gruppe, sonst -1.

        `vals` ist pos bei Treffer und -1 sonst. Weil Positionen streng steigen,
        liegt jeder Treffer einer FRUEHEREN Gruppe unter dem Gruppenanfang —
        ein einziges maximum.accumulate plus Vergleich mit dem Gruppenanfang
        reicht als Zuruecksetzung.
        """
        shifted = np.empty(n, np.int64)
        shifted[0] = -1
        shifted[1:] = vals[:-1]
        run = np.maximum.accumulate(shifted)
        return np.where(run >= start, run, -1)

    last_aggr = cummax_prev(np.where(is_aggr, pos, -1), start_hs)
    # Sitz des laufenden (Praeflop-)Aggressors
    pre = a_str == 0
    la_pre = cummax_prev(np.where(is_aggr & pre, pos, -1), start_h)
    # letzter Praeflop-Aggressor der Hand (fuer Postflop-Aktionen legal: die
    # Praeflop-Runde ist zum Zeitpunkt einer Postflop-Aktion abgeschlossen)
    hmax = int(hloc.max()) + 1 if n else 1
    pfa_pos = np.full(hmax, -1, np.int64)
    m = is_aggr & pre
    if m.any():
        np.maximum.at(pfa_pos, hloc[m], pos[m])
    pfa_seat_h = np.where(pfa_pos >= 0, a_seat[np.maximum(pfa_pos, 0)], -1)
    cur_seat = np.where(la_pre >= 0, a_seat[np.maximum(la_pre, 0)], -1)
    is_pfa = np.where(pre, a_seat == cur_seat,
                      a_seat == pfa_seat_h[hloc]).astype(np.float32)

    # Fold-Position je (Hand, Sitz); Sitz gilt ab da als tot
    fold_pos = np.full((hmax, 6), np.iinfo(np.int64).max, np.int64)
    mf = a_act == A_FOLD
    if mf.any():
        fold_pos[hloc[mf], a_seat[mf]] = pos[mf]     # je (Hand,Sitz) hoechstens einer
    # letzte eigene Aktion je Sitz VOR der laufenden (in derselben Setzrunde)
    n_to_act = np.zeros(n, np.float32)
    n_behind = np.zeros(n, np.float32)
    first = np.where(a_str == 0, (btn_h[hloc] + 3) % 6, (btn_h[hloc] + 1) % 6)
    my_ord = (a_seat - first) % 6
    for s in range(6):
        alive = fold_pos[hloc, s] > pos
        other = a_seat != s
        last_s = cummax_prev(np.where(a_seat == s, pos, -1), start_hs)
        must = alive & other & (last_s <= last_aggr)
        s_ord = (s - first) % 6
        behind = alive & other & (s_ord > my_ord)
        n_to_act += (must & behind).astype(np.float32)
        n_behind += behind.astype(np.float32)
    return n_to_act, n_behind, is_pfa


# ---------------------------------------------------------------------------
# 3. vollstaendige Merkmalsmatrix
# ---------------------------------------------------------------------------
def features(hsel: np.ndarray):
    """(X (n, N_COLS), cls (n,), act_index (n,)) fuer die Haende `hsel`.

    `hsel` muss zu EINER (Tisch, Phase)-Einheit gehoeren (wie in policy.py).
    """
    X0, cls = engine.policy_features(hsel)
    assert X0.shape[1] == 22, f"erwartet 22 Basisspalten, bekommen {X0.shape[1]}"
    off = np.asarray(A("act_offsets"))
    lens = off[hsel + 1] - off[hsel]
    sel = np.concatenate([np.arange(off[h], off[h + 1]) for h in hsel])
    hloc = np.repeat(np.arange(len(hsel), dtype=np.int64), lens)
    a_seat = np.asarray(A("act_seat"))[sel].astype(np.int64)
    a_act = np.asarray(A("act_action"))[sel]
    a_str = np.asarray(A("act_street"))[sel].astype(np.int64)
    a_tc = np.asarray(A("act_to_call"))[sel].astype(np.float32)
    a_pot = np.asarray(A("act_pot_before"))[sel].astype(np.float32)
    a_amt = np.asarray(A("act_amount"))[sel].astype(np.float32)
    btn = np.asarray(A("hand_button_seat"))[hsel].astype(np.int64)
    bb = np.asarray(A("hand_bb"))[hsel].astype(np.float32)
    is_aggr = (a_act == engine.A_BET) | (a_act == engine.A_RAISE) | (
        (a_act == engine.A_ALLIN) & (a_amt > a_tc))
    nta, nbh, pfa = order_block(hloc, a_str, a_seat, a_act, btn, is_aggr)

    # --- Spielergewohnheiten mit Leave-one-hand-out --------------------------
    ACT_TOT, HND_TOT, _, _ = load_stats()
    sp = np.asarray(A("seat_player")).astype(np.int64)
    hp = np.asarray(A("hand_phase")).astype(np.int64)
    ph = int(hp[hsel[0]])
    assert (hp[hsel] == ph).all(), "features() erwartet genau eine Phase"
    pl_hand = sp[hsel]                                     # (H, 6)
    pl_act = pl_hand[hloc, a_seat]
    g_act = pl_act * 2 + ph
    C = _act_contrib(a_str, a_act, a_tc, a_pot, a_amt, bb[hloc])
    # Beitrag der LAUFENDEN Hand je (Hand, Sitz) aufsummieren
    hs = hloc * 6 + a_seat
    own_a = np.zeros((len(hsel) * 6, KA), np.float64)
    for k in range(KA):
        own_a[:, k] = np.bincount(hs, weights=C[:, k].astype(np.float64),
                                  minlength=len(hsel) * 6)
    del C
    act_num = ACT_TOT[g_act] - own_a[hs]
    # Handebene: die laufende Hand traegt hoechstens 1 zu jedem Zaehler bei
    folded = np.asarray(A("seat_folded"))[hsel]
    own_h = np.zeros((len(hsel) * 6, KH), np.float64)
    own_h[:, 0] = 1.0
    own_h[:, 4] = (~folded).reshape(-1).astype(np.float64)
    pre = a_str == 0
    is_vol = (a_act == A_CALL) | is_aggr
    for col, mask in ((1, pre), (2, pre & is_vol), (3, pre & is_aggr)):
        u = np.unique(hs[mask])
        own_h[u, col] = 1.0
    # Sitze ohne Aktion in dieser Hand: own_h[:,0] muss trotzdem 1 sein (der
    # Spieler sass am Tisch) — das ist oben bereits gesetzt.
    hnd_num = HND_TOT[g_act] - own_h[hs]
    np.maximum(act_num, 0.0, out=act_num)
    np.maximum(hnd_num, 0.0, out=hnd_num)
    P = _rates(act_num, hnd_num)

    X = np.empty((len(sel), N_COLS), np.float32)
    X[:, :22] = X0
    X[:, 22] = nta
    X[:, 23] = nbh
    X[:, 24] = pfa
    X[:, 25:] = P
    return X, cls, sel
