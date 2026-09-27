"""Tisch-paralleler Merkmalsmotor.

Struktur der Daten: 400 persistente Pools zu je 30 Spielern, jeder Pool an genau
einem Tisch. Spieler wechseln nie den Pool. Daraus folgt: ein Tisch ist eine
vollstaendig abgeschlossene Rechen- UND Validierungseinheit.

Je (Tisch, Phase) werden berechnet:
  1. per (Hand, Sitz)  — Aktions- und Kartenzusammenfassungen
  2. per (Hand, Sitz_a, Sitz_b) — gerichtete Konfrontationszaehler
  3. per (Paar, Hand)  — relationale Merkmale ("PH-Merkmale")
  4. per Paar          — Aggregate: Mittel, Maximum, Top-k, Quantil, bestes
                         Zeitfenster + spielerbereinigte Kontraste

Alle gerichteten Groessen werden ordnungsinvariant symmetrisiert (Summe,
Betragsdifferenz, Max, Min, Asymmetrie), damit die willkuerliche Reihenfolge
innerhalb eines Paares kein Merkmal wird.
"""
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import numpy as np

import handeval

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("POKER_CACHE", str(ROOT / "cache")))
NP_PLAYERS = 12000
COMBOS = list(combinations(range(6), 2))
EPS = 1e-9

# Aktionscodes (siehe prep.py)
A_FOLD, A_CHECK, A_CALL, A_BET, A_RAISE, A_ALLIN = 0, 1, 2, 3, 4, 5
AGGR = (A_BET, A_RAISE, A_ALLIN)

_ARR: dict[str, np.ndarray] = {}

# ---------------------------------------------------------------------------
# Situationsbereinigung: empirische Erwartung je Situationsklasse
# ---------------------------------------------------------------------------
# Der Vergleich "hat der Spieler hier ungewoehnlich gehandelt" braucht eine
# Referenz fuer die SITUATION, nicht nur fuer den Spieler. Die Referenz wird
# rein empirisch aus allen 18,6 Mio Aktionen aller 12.000 Spieler gebildet:
#   Situationsklasse = (Street, Pot-Odds-Stufe, aktive Spieler, Blattstufe, Position)
# Es fliessen KEINE Labels ein, und der Beitrag eines einzelnen Paares zu einer
# Klasse mit im Mittel ~6.000 Aktionen ist vernachlaessigbar; die Klassen sind
# damit gegenueber den Validierungsfolds praktisch isoliert. Zusaetzlich wird
# der eigene Beitrag jeder Aktion herausgerechnet (Leave-one-out).
N_ODDS, N_ACT_PL, N_STR, N_POS = 5, 5, 5, 6
N_BINS = 4 * N_ODDS * N_ACT_PL * N_STR * N_POS
_EXPECT = None


def action_bin(street, to_call, pot_before, players_active, dec_cat, pre_pct, position):
    """Situationsklasse je Aktion (int in [0, N_BINS))."""
    odds = np.where(to_call <= 0, 0,
                    1 + np.clip(np.digitize(to_call / np.maximum(pot_before + to_call, 1.0),
                                            [0.20, 0.30, 0.40]), 0, 3))
    npl = np.clip(players_active, 2, 6) - 2
    strb = np.where(street == 0,
                    np.clip((pre_pct * N_STR).astype(np.int64), 0, N_STR - 1),
                    np.clip(dec_cat.astype(np.int64), 0, N_STR - 1))
    pos = np.clip(position, 0, N_POS - 1)
    return (((street * N_ODDS + odds) * N_ACT_PL + npl) * N_STR + strb) * N_POS + pos


def load_expect():
    """(N_BINS, 4)-Tabelle mit P(aggressiv|fold|call|check) je Situationsklasse."""
    global _EXPECT
    if _EXPECT is None:
        f = CACHE / "expect_table.npy"
        _EXPECT = np.load(f) if f.exists() else None
    return _EXPECT


def A(name: str) -> np.ndarray:
    if name not in _ARR:
        _ARR[name] = np.load(CACHE / f"{name}.npy", mmap_mode="r")
    return _ARR[name]


# --------------------------------------------------------------------------
# Preflop-Handstaerke (Chen-Formel) — 169 Startkartenklassen
# --------------------------------------------------------------------------
def _chen_table() -> np.ndarray:
    """chen[c1, c2] fuer Kartencodes 0..51; -1 wenn eine Karte fehlt."""
    pts = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10])  # 2..A
    tab = np.zeros((52, 52), dtype=np.float32)
    r = np.arange(52) // 4
    s = np.arange(52) % 4
    for c1 in range(52):
        hi = np.maximum(r[c1], r)
        lo = np.minimum(r[c1], r)
        v = pts[hi].copy()
        pair = r[c1] == r
        v = np.where(pair, np.maximum(2 * pts[hi], 5.0), v)
        v = v + np.where((s[c1] == s) & ~pair, 2.0, 0.0)
        gap = hi - lo - 1
        pen = np.select([gap <= 0, gap == 1, gap == 2, gap == 3],
                        [0.0, 1.0, 2.0, 4.0], default=5.0)
        v = v - np.where(pair, 0.0, pen)
        v = v + np.where((~pair) & (gap <= 1) & (hi < 10), 1.0, 0.0)  # beide < Q
        tab[c1] = np.ceil(np.maximum(v, 0.0))
    return tab


CHEN = _chen_table()
_CHEN_FLAT = CHEN.ravel()
# Perzentil der Chen-Punktzahl ueber alle 1326 Starthaende (gleichgewichtet)
_pairs = np.array([(i, j) for i in range(52) for j in range(i + 1, 52)])
_vals = CHEN[_pairs[:, 0], _pairs[:, 1]]
# Original pure-card tie order, independent of NumPy CPU dispatch.
_chen_order_file = os.environ.get("POKER_CHEN_ORDER_FILE")
if _chen_order_file:
    import json as _chen_json
    _chen_spec = _chen_json.loads(Path(_chen_order_file).read_text())
    _order = np.asarray(_chen_spec["order"], dtype=np.int64)
    assert _order.shape == (len(_vals),)
    assert np.array_equal(np.sort(_order), np.arange(len(_vals)))
    assert np.all(np.diff(_vals[_order]) >= 0)
else:
    _order = np.argsort(_vals)
_pct = np.empty(len(_vals))
_pct[_order] = np.linspace(0, 1, len(_vals))
CHEN_PCT = np.zeros((52, 52), dtype=np.float32)
CHEN_PCT[_pairs[:, 0], _pairs[:, 1]] = _pct
CHEN_PCT[_pairs[:, 1], _pairs[:, 0]] = _pct
_CHEN_PCT_FLAT = CHEN_PCT.ravel()
if _chen_order_file:
    import hashlib as _chen_hashlib
    assert _chen_hashlib.sha256(CHEN_PCT.tobytes()).hexdigest() == _chen_spec["table_sha256"]


def chen_points(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Rohe Chen-Punktzahl der Startkarten (fehlende Karte -> 0)."""
    ok = (h1 >= 0) & (h2 >= 0)
    idx = np.where(ok, h1.astype(np.int32) * 52 + h2.astype(np.int32), 0)
    return np.where(ok, _CHEN_FLAT[idx], 0.0).astype(np.float32)


def preflop_strength(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Perzentil der Startkartenstaerke, 0..1. Fehlende Karte -> 0.5."""
    ok = (h1 >= 0) & (h2 >= 0)
    idx = np.where(ok, h1.astype(np.int32) * 52 + h2.astype(np.int32), 0)
    return np.where(ok, _CHEN_PCT_FLAT[idx], 0.5).astype(np.float32)


# --------------------------------------------------------------------------
# Aktionsebene: letzter Aggressor vor jeder Aktion
# --------------------------------------------------------------------------
def last_aggressor(hloc: np.ndarray, street: np.ndarray, seat: np.ndarray,
                   action: np.ndarray, amount: np.ndarray, to_call: np.ndarray,
                   bb_seat_of_hand: np.ndarray) -> np.ndarray:
    """Sitz des letzten Aggressors VOR jeder Aktion (-1 = keiner).

    Aggression = bet, raise, oder all_in mit amount > to_call. Preflop gilt der
    Big Blind als initialer Aggressor (die Blinds stehen nicht im Aktionslog).
    """
    n = len(hloc)
    is_aggr = (action == A_BET) | (action == A_RAISE) | ((action == A_ALLIN) & (amount > to_call))
    idx = np.arange(n)
    x = np.where(is_aggr, idx, -1)
    cm = np.maximum.accumulate(x)                     # letzter Aggressor inkl. self
    cm_prev = np.empty(n, dtype=np.int64)
    cm_prev[0] = -1
    cm_prev[1:] = cm[:-1]
    # Gruppengrenzen (Hand, Street): Aggressor darf nicht ueber sie hinweg gelten
    grp_new = np.empty(n, dtype=bool)
    grp_new[0] = True
    grp_new[1:] = (hloc[1:] != hloc[:-1]) | (street[1:] != street[:-1])
    grp_start = np.maximum.accumulate(np.where(grp_new, idx, 0))
    valid = cm_prev >= grp_start
    out = np.where(valid, seat[np.maximum(cm_prev, 0)], -1).astype(np.int8)
    # Preflop ohne vorherigen Aggressor -> Big Blind
    pre = (street == 0) & (~valid)
    out[pre] = bb_seat_of_hand[hloc[pre]]
    # der BB selbst hat keinen Gegner-Aggressor
    out[pre & (seat == out)] = -1
    return out


# --------------------------------------------------------------------------
# Kernberechnung je (Tisch, Phase)
# --------------------------------------------------------------------------
PH_NAMES: list[str] = []


def table_pair_hands(hsel: np.ndarray):
    """PH-Merkmale fuer alle 15 Sitzpaare aller Haende in `hsel` (zeitsortiert).

    Rueckgabe: (pkey, hidx, torder (chronologischer Rang der Hand innerhalb
                Tisch+Phase), X (n, F) float32, names)
    """
    H = len(hsel)
    bb = A("hand_bb")[hsel].astype(np.float32)
    pot = A("hand_final_pot")[hsel].astype(np.float32)
    btn = A("hand_button_seat")[hsel].astype(np.int8)
    nsd = A("hand_n_showdown")[hsel].astype(np.float32)
    nboard = A("hand_n_board")[hsel].astype(np.float32)

    sp = A("seat_player")[hsel].astype(np.int32)          # (H,6)
    contrib = A("seat_contrib")[hsel].astype(np.float32)
    net = A("seat_net")[hsel].astype(np.float32)
    stack = A("seat_stack")[hsel].astype(np.float32)
    folded = A("seat_folded")[hsel]
    showd = A("seat_showdown")[hsel]
    wshare = A("seat_won_share")[hsel].astype(np.float32)
    h1 = A("seat_hole1")[hsel].astype(np.int16)
    h2 = A("seat_hole2")[hsel].astype(np.int16)

    # ---- forensische Handstaerken -------------------------------------
    # str_lev[l] = Staerke aller sechs Sitze bei l sichtbaren Boardkarten
    # (l = 0 preflop, 3 Flop, 4 Turn, 5 River). Fehlende Karten sind -1 und
    # werden von handeval ignoriert. Die Staerke wird NUR forensisch benutzt.
    board = A("hand_board")[hsel].astype(np.int16)
    holes = np.stack([h1, h2], axis=2)                     # (H,6,2)
    str_lev = np.empty((4, H, 6), dtype=np.int64)
    for li, nc in enumerate((0, 3, 4, 5)):
        if nc == 0:
            cc = holes
        else:
            cc = np.concatenate([holes, np.broadcast_to(board[:, None, :nc], (H, 6, nc))], axis=2)
        str_lev[li] = handeval.evaluate(cc.reshape(-1, cc.shape[2])).reshape(H, 6)
    cat_lev = (str_lev // (handeval.B ** 6)).astype(np.int8)      # 0..8 Handkategorie
    nb_lev = np.stack([(str_lev[l][:, None, :] > str_lev[l][:, :, None]).sum(2)
                       for l in range(4)]).astype(np.int8)         # (4,H,6)
    str_fin = str_lev[3]
    # Rang ohne Sitzreihenfolge-Abhaengigkeit: Anzahl echt staerkerer Sitze
    n_better = (str_fin[:, None, :] > str_fin[:, :, None]).sum(2).astype(np.float32)
    is_best = (n_better == 0).astype(np.float32)

    bbc = bb[:, None]
    contrib_bb = contrib / bbc
    net_bb = net / bbc
    stack_bb = stack / bbc
    pot_bb = pot / bb
    pos = (np.arange(6)[None, :] - btn[:, None]) % 6      # 0=Button,1=SB,2=BB,...
    pre_str = preflop_strength(h1, h2)                    # (H,6) 0..1
    chen = chen_points(h1, h2)                            # (H,6) Chen-Punkte

    # ---- Aktionen dieser Haende einsammeln ----
    off = A("act_offsets")
    lens = off[hsel + 1] - off[hsel]
    sel = np.concatenate([np.arange(off[h], off[h + 1]) for h in hsel]) if H else np.zeros(0, np.int64)
    hloc = np.repeat(np.arange(H, dtype=np.int32), lens)
    a_seat = A("act_seat")[sel].astype(np.int8)
    a_act = A("act_action")[sel].astype(np.int8)
    a_str = A("act_street")[sel].astype(np.int8)
    a_amt = A("act_amount")[sel].astype(np.float32)
    a_amtto = A("act_amount_to")[sel].astype(np.float32)
    a_pot = A("act_pot_before")[sel].astype(np.float32)
    a_tc = A("act_to_call")[sel].astype(np.float32)
    a_stack = A("act_stack_before")[sel].astype(np.float32)
    a_pa = A("act_players_active")[sel].astype(np.int8)
    a_bb = bb[hloc]

    bb_seat = (btn + 2) % 6
    aggr_of = last_aggressor(hloc, a_str, a_seat, a_act, a_amt, a_tc, bb_seat)

    is_aggr = (a_act == A_BET) | (a_act == A_RAISE) | ((a_act == A_ALLIN) & (a_amt > a_tc))
    is_fold = a_act == A_FOLD
    is_call = (a_act == A_CALL) | ((a_act == A_ALLIN) & (a_amt <= a_tc))
    is_check = a_act == A_CHECK
    faces = a_tc > 0

    # ---- (Hand, Sitz) Aggregate ----
    # np.add.at ist hier der Flaschenhals; np.bincount ist rund 20x schneller
    hs_idx = hloc.astype(np.int64) * 6 + a_seat
    HS6 = H * 6

    def hs(vals, mask=None):
        v = vals if mask is None else np.where(mask, vals, 0.0)
        return np.bincount(hs_idx, weights=v, minlength=HS6).reshape(H, 6).astype(np.float32)

    one = np.ones(len(hloc), dtype=np.float32)
    s_nact = hs(one)
    s_aggr = hs(one, is_aggr)
    s_fold = hs(one, is_fold)
    s_call = hs(one, is_call)
    s_check = hs(one, is_check)
    s_faced = hs(one, faces)
    s_foldfaced = hs(one, is_fold & faces)
    s_aggr_pre = hs(one, is_aggr & (a_str == 0))
    s_vpip = hs(one, (a_str == 0) & (is_call | is_aggr))
    s_invest = hs(a_amt / a_bb)
    s_laststreet = np.zeros((H, 6), dtype=np.int64)
    np.maximum.at(s_laststreet, (hloc, a_seat), a_str.astype(np.int64))
    # Staerke zum Zeitpunkt der letzten Aktion (nur sichtbare Karten)
    str_dec = np.take_along_axis(str_lev, s_laststreet[None, :, :], axis=0)[0]
    n_better_dec = (str_dec[:, None, :] > str_dec[:, :, None]).sum(2).astype(np.float32)
    s_maxbet = np.zeros((H, 6), dtype=np.float32)
    np.maximum.at(s_maxbet, (hloc, a_seat), np.where(is_aggr, a_amtto / a_bb, 0.0))

    # ---- (Hand, Akteur, Aggressor) gerichtete Zaehler ----
    ok = aggr_of >= 0
    ii, jj = hloc[ok], a_seat[ok]
    kk = aggr_of[ok]

    d3_idx = (ii.astype(np.int64) * 6 + jj) * 6 + kk
    D36 = H * 36

    def dir3(mask):
        return np.bincount(d3_idx, weights=mask[ok].astype(np.float64),
                           minlength=D36).reshape(H, 6, 6).astype(np.float32)

    d_any = dir3(np.ones(len(hloc), bool))
    d_faced = dir3(faces)
    d_fold = dir3(is_fold & faces)
    d_call = dir3(is_call & faces)
    d_reraise = dir3(is_aggr & faces)          # ueber die Aggression erhoeht
    d_callamt = np.zeros((H, 6, 6), dtype=np.float32)
    np.add.at(d_callamt, (ii, jj, kk), np.where(is_call[ok], (a_amt / a_bb)[ok], 0.0))
    # gefoldete Handstaerke gegen diesen Aggressor
    d_foldstr = np.zeros((H, 6, 6), dtype=np.float32)
    np.add.at(d_foldstr, (ii, jj, kk), np.where((is_fold & faces)[ok], pre_str[hloc, a_seat][ok], 0.0))
    d_callstr = np.zeros((H, 6, 6), dtype=np.float32)
    np.add.at(d_callstr, (ii, jj, kk), np.where(is_call[ok] & faces[ok], pre_str[hloc, a_seat][ok], 0.0))

    # ---- Entscheidungs-Equity gegen den Aggressor -----------------------
    # Vorberechnet in equity_apply.py: Equity des Handelnden gegen das
    # tatsaechliche Blatt des letzten Aggressors, mit dem damals sichtbaren
    # Board. Rein forensisch — der Spieler kennt diese Karten nicht, und genau
    # deshalb ist eine systematische Abweichung verdaechtig.
    if (CACHE / "act_equity.npy").exists():
        a_eq = A("act_equity")[sel].astype(np.float32)
    else:
        a_eq = np.full(len(hloc), np.nan, dtype=np.float32)
    eq_ok = np.isfinite(a_eq)
    eq0 = np.nan_to_num(a_eq, nan=0.0)

    # ---- Entscheidungsqualitaet je Aktion ------------------------------
    # Bewertet wird die Entscheidung mit den ZUM AKTIONSZEITPUNKT sichtbaren
    # Karten (eigene Hole Cards + bereits gezeigtes Board). Ob die Entscheidung
    # gegen die Karten der anderen lief, ist forensisch — der Spieler konnte es
    # nicht wissen, und genau das macht die Abweichung verdaechtig.
    a_lev = a_str.astype(np.int64)
    dec_nb = nb_lev[a_lev, hloc, a_seat].astype(np.float32)      # bessere Blaetter
    dec_cat = cat_lev[a_lev, hloc, a_seat].astype(np.float32)    # Handkategorie
    dec_best = (dec_nb == 0).astype(np.float32)
    pot_odds = a_tc / np.maximum(a_pot + a_tc, 1.0)
    # Direkter Zweikampfvergleich zum Aktionszeitpunkt: war der Foldende
    # gegenueber GENAU DEM AGGRESSOR vorn? Das ist schaerfer als der Vergleich
    # gegen alle sechs Sitze, weil bei einem Fold gegen den Partner nur dieser
    # Zweikampf zaehlt. Gemessen an directed_transfer: die Equity des Gebers
    # gegen die echten Partnerkarten liegt im Moment des Folds im Median bei
    # 77 % gegen 30 % in gewoehnlichen gemeinsamen Haenden.
    opp = np.maximum(aggr_of, 0).astype(np.int64)
    dec_str_self = str_lev[a_lev, hloc, a_seat]
    dec_str_opp = str_lev[a_lev, hloc, opp]
    ahead = (dec_str_self > dec_str_opp) & (aggr_of >= 0)
    q_foldahead = np.where(is_fold & faces & ahead, 1.0, 0.0)
    q_foldahead_cheap = q_foldahead * (1.0 - pot_odds)
    q_foldahead_cat = np.where(is_fold & faces & ahead, dec_cat, 0.0)
    q_callbehind = np.where(is_call & faces & ~ahead, (a_amt / a_bb), 0.0)

    # bestes Blatt billig weggeworfen -> maximal verdaechtig
    q_foldbest = np.where(is_fold & faces, dec_best * (1.0 - pot_odds), 0.0)
    q_foldcat = np.where(is_fold & faces, dec_cat, 0.0)
    q_foldmade = np.where(is_fold & faces, (dec_cat >= 1) * (1.0 - pot_odds), 0.0)
    # mit dem schlechtesten Blatt teuer mitgegangen
    q_callworst = np.where(is_call & faces, (dec_nb / 5.0) * pot_odds * (a_amt / a_bb), 0.0)
    q_checkbest = np.where(is_check, dec_best, 0.0)

    def dirmax(vals):
        out = np.zeros(H * 36, dtype=np.float32)
        np.maximum.at(out, d3_idx, vals[ok].astype(np.float32))
        return out.reshape(H, 6, 6)

    # Die Hand endet mit einem Fold des einen Partners gegen den anderen.
    # In 75 % der directed_transfer-Evidenzhaende ist das der Abschluss.
    last_idx = np.cumsum(lens) - 1
    last_ok = ok[last_idx] & is_fold[last_idx]
    d_lastfold = np.zeros((H, 6, 6), dtype=np.float32)
    li = np.flatnonzero(last_ok)
    if len(li):
        d_lastfold[li, a_seat[last_idx][li], aggr_of[last_idx][li]] = 1.0

    # Street-aufgeloeste Konfrontation: derselbe Fold wiegt auf dem River
    # anders als praeflop, weil dort schon investiert wurde.
    d_fold_st = np.zeros((4, H, 6, 6), dtype=np.float32)
    d_aggr_st = np.zeros((4, H, 6, 6), dtype=np.float32)
    for st_ in range(4):
        m_ = ok & (a_str == st_)
        if m_.any():
            idx_ = (hloc[m_].astype(np.int64) * 6 + a_seat[m_]) * 6 + aggr_of[m_]
            d_fold_st[st_] = np.bincount(idx_, weights=(is_fold & faces)[m_].astype(np.float64),
                                         minlength=H * 36).reshape(H, 6, 6)
            d_aggr_st[st_] = np.bincount(idx_, weights=is_aggr[m_].astype(np.float64),
                                         minlength=H * 36).reshape(H, 6, 6)

    # Preis, zu dem gegen den Partner gefoldet wurde
    d_foldodds = dirmax(np.where(is_fold & faces, pot_odds, 0.0))
    d_foldtocall = dirmax(np.where(is_fold & faces, a_tc / a_bb, 0.0))
    d_foldinvested = dirmax(np.where(is_fold & faces, a_pot / a_bb, 0.0))

    ff = is_fold & faces & eq_ok
    cc = is_call & faces & eq_ok
    d_eqf_sum = dir3(eq0 * ff)
    d_eqf_n = dir3(ff)
    d_eqc_sum = dir3(eq0 * cc)
    d_eqc_n = dir3(cc)
    d_eqf_max = dirmax(np.where(ff, eq0, 0.0))
    d_eqf_val = dirmax(np.where(ff, eq0 * (a_pot / a_bb), 0.0))
    s_eqf_sum = hs(eq0 * ff); s_eqf_n = hs(ff.astype(np.float32))
    s_eqc_sum = hs(eq0 * cc); s_eqc_n = hs(cc.astype(np.float32))

    d_qfoldbest = dirmax(q_foldbest)
    d_qfoldahead = dirmax(q_foldahead)
    d_qfoldahead_cheap = dirmax(q_foldahead_cheap)
    d_qfoldahead_cat = dirmax(q_foldahead_cat)
    d_qcallbehind = dirmax(q_callbehind)
    d_qfoldcat = dirmax(q_foldcat)
    d_qfoldmade = dirmax(q_foldmade)
    d_qcallworst = dirmax(q_callworst)
    s_checkbest = hs(np.ones(len(hloc), np.float32), is_check & (dec_best > 0))

    # ---- situationsbereinigte Residuen -----------------------------------
    EXP = load_expect()
    a_pos = ((a_seat.astype(np.int64) - btn[hloc]) % 6)
    a_bin = action_bin(a_str, a_tc, a_pot, a_pa, dec_cat, pre_str[hloc, a_seat], a_pos)
    if EXP is not None:
        e = EXP[a_bin]                                   # (n,4) erwartete Anteile
        r_aggr = is_aggr.astype(np.float32) - e[:, 0]
        r_fold = is_fold.astype(np.float32) - e[:, 1]
        r_call = is_call.astype(np.float32) - e[:, 2]
        r_check = is_check.astype(np.float32) - e[:, 3]
    else:
        r_aggr = r_fold = r_call = r_check = np.zeros(len(hloc), dtype=np.float32)

    # ---- Ueberraschung je Aktion aus dem Policy-Modell --------------------
    # -log P(beobachtete Aktion | Situation). Die Wettbewerbsbeschreibung sagt,
    # jede geplante Evidenzhand enthalte eine "behavior-specific action visible
    # in the public action log". Genau das misst die Ueberraschung: eine
    # Entscheidung, die unter der normalen Handlungsverteilung unwahrscheinlich
    # ist. Das Policy-Modell kennt weder Labels noch Spieleridentitaeten.
    # Die Ueberraschung wird einmalig von policy_apply.py fuer alle Aktionen
    # berechnet und hier nur indiziert — kein Modelllauf im Arbeiterprozess.
    _sfx = (POLICY_Q2_SFX if POLICY_Q2 else
            ("_v2" if POLICY_V2 else
             ("_pfs" if (POLICY_FIX and SURP_STRAT) else ("_pf" if POLICY_FIX else ""))))
    if (CACHE / f"act_surprise{_sfx}.npy").exists():
        surp = A(f"act_surprise{_sfx}")[sel].astype(np.float32)
        p_aggr_hat = A(f"act_paggr{_sfx}")[sel].astype(np.float32)
        p_fold_hat = A(f"act_pfold{_sfx}")[sel].astype(np.float32)
    else:
        surp = np.zeros(len(hloc), dtype=np.float32)
        p_aggr_hat = p_fold_hat = np.zeros(len(hloc), dtype=np.float32)
    m_aggr_dev = is_aggr.astype(np.float32) - p_aggr_hat
    m_fold_dev = is_fold.astype(np.float32) - p_fold_hat
    if ENGINE_SZ and (CACHE / "act_sizesurp.npy").exists():
        sz_surp = A("act_sizesurp")[sel].astype(np.float32)
        sz_dev = A("act_sizedev")[sel].astype(np.float32)
    else:
        sz_surp = sz_dev = np.zeros(len(hloc), dtype=np.float32)

    # ---- Verhalten bedingt darauf, ob der PARTNER NOCH IM POT IST --------
    # Die schaerfste bekannte Signatur von soft_play ist kein Spielertrait,
    # sondern ein Kontrast INNERHALB derselben Haende: solange der Partner noch
    # lebt, ersetzt der Spieler Fold und Raise durch den Flat Call; sobald der
    # Partner gefoldet hat, spielt derselbe Spieler wie jeder andere.
    # Deshalb je (Hand, Akteur, anderer Sitz) zaehlen, ob der andere Sitz zum
    # Aktionszeitpunkt noch nicht gefoldet hatte.
    ano = np.concatenate([np.arange(l) for l in lens]) if H else np.zeros(0, np.int64)
    fold_ano = np.full((H, 6), np.iinfo(np.int32).max, dtype=np.int64)
    np.minimum.at(fold_ano, (hloc[is_fold], a_seat[is_fold]), ano[is_fold])
    alive = fold_ano[hloc] >= ano[:, None]                    # (n_act, 6)

    al_idx = ((hloc.astype(np.int64) * 6 + a_seat)[:, None] * 6 + np.arange(6)).ravel()
    if ENGINE_CF and (CACHE / "act_cfto.npy").exists():
        _cfs = A("act_cfself")[sel].astype(np.float32)
        _cft = np.asarray(A("act_cfto")[sel], dtype=np.float32)        # (n, 5), Versatz 1..5
        # Versatz -> absoluter Sitz: other = (Akteur + Versatz) % 6
        _seatmat = np.zeros((len(hloc), 6), np.float32)
        _rows = np.arange(len(hloc))
        for _o in range(1, 6):                                          # Versatz 1..5
            _seatmat[_rows, (a_seat.astype(np.int64) + _o) % 6] = _cft[:, _o - 1]
        CFTO = np.bincount(al_idx, weights=_seatmat.astype(np.float64).ravel(),
                           minlength=H * 36).reshape(H, 6, 6).astype(np.float32)
        CFSELF = np.bincount(hloc.astype(np.int64) * 6 + a_seat,
                             weights=_cfs.astype(np.float64),
                             minlength=H * 6).reshape(H, 6).astype(np.float32)
    else:
        CFTO = np.zeros((H, 6, 6), np.float32)
        CFSELF = np.zeros((H, 6), np.float32)

    def dir_alive(mask):
        """(H, Akteur, anderer) Zaehler nur fuer Aktionen mit lebendem anderen."""
        w = (mask.astype(np.float64)[:, None] * alive).ravel()
        return np.bincount(al_idx, weights=w, minlength=H * 36).reshape(H, 6, 6).astype(np.float32)

    pre = a_str == 0
    post = ~pre
    weak = pre_str[hloc, a_seat] < 0.34                       # schwache Startkarten
    PA = {
        "pre_n": dir_alive(pre), "pre_call": dir_alive(pre & is_call),
        "pre_fold": dir_alive(pre & is_fold), "pre_aggr": dir_alive(pre & is_aggr),
        "prew_n": dir_alive(pre & weak), "prew_call": dir_alive(pre & weak & is_call),
        "prew_fold": dir_alive(pre & weak & is_fold),
        "post_n": dir_alive(post), "post_check": dir_alive(post & is_check),
        "post_call": dir_alive(post & is_call), "post_aggr": dir_alive(post & is_aggr),
        "post_fold": dir_alive(post & is_fold),
        "bestpass": dir_alive(post & (dec_best > 0) & (is_check | is_call)),
        "bestn": dir_alive(post & (dec_best > 0)),
    }
    # Residuensummen (nicht 0/1, deshalb eigener Weg)
    def dir_alive_val(vals):
        w = (vals.astype(np.float64)[:, None] * alive).ravel()
        return np.bincount(al_idx, weights=w, minlength=H * 36).reshape(H, 6, 6).astype(np.float32)

    # Aggression mit dem schlechtesten sichtbaren Blatt, gewichtet mit der
    # Einsatzhoehe: baut Pots auf, ohne auf das eigene Blatt zu setzen.
    q_aggrworst = np.where(is_aggr, (dec_nb / 5.0) * (a_amt / a_bb), 0.0)
    RECK = dir_alive_val(q_aggrworst)
    RECKT = hs(q_aggrworst)

    SURP = {"s_all": dir_alive_val(surp),
            "s_fold": dir_alive_val(np.where(is_fold, surp, 0.0)),
            "s_aggr": dir_alive_val(np.where(is_aggr, surp, 0.0)),
            "s_pass": dir_alive_val(np.where(is_call | is_check, surp, 0.0)),
            "m_aggr": dir_alive_val(m_aggr_dev),
            "m_fold": dir_alive_val(m_fold_dev)}
    if ENGINE_SZ:
        # drei Kanaele: WIE UNGEWOEHNLICH war die Hoehe, in WELCHE RICHTUNG,
        # und wie stark ohne Richtung. Der Richtungskanal ist der eigentliche
        # Punkt - er trennt "zu klein" von "zu gross".
        SURP["z_szs"] = dir_alive_val(sz_surp)
        SURP["z_szd"] = dir_alive_val(sz_dev)
        SURP["z_sza"] = dir_alive_val(np.abs(sz_dev))
    SURPT = {"s_all": hs(surp), "s_fold": hs(np.where(is_fold, surp, 0.0)),
             "s_aggr": hs(np.where(is_aggr, surp, 0.0)),
             "s_pass": hs(np.where(is_call | is_check, surp, 0.0)),
             "m_aggr": hs(m_aggr_dev), "m_fold": hs(m_fold_dev)}
    if ENGINE_SZ:
        SURPT["z_szs"] = hs(sz_surp)
        SURPT["z_szd"] = hs(sz_dev)
        SURPT["z_sza"] = hs(np.abs(sz_dev))
    d_surpmax = dirmax(surp)

    RES = {"r_aggr": dir_alive_val(r_aggr), "r_fold": dir_alive_val(r_fold),
           "r_call": dir_alive_val(r_call), "r_check": dir_alive_val(r_check),
           "r_aggr_str": dir_alive_val(r_aggr * (pre_str[hloc, a_seat] - 0.5)),
           "r_aggr_pot": dir_alive_val(r_aggr * np.log1p(a_pot / a_bb))}
    RTOT = {"r_aggr": hs(r_aggr), "r_fold": hs(r_fold), "r_call": hs(r_call),
            "r_check": hs(r_check),
            "r_aggr_str": hs(r_aggr * (pre_str[hloc, a_seat] - 0.5)),
            "r_aggr_pot": hs(r_aggr * np.log1p(a_pot / a_bb))}
    TOT = {k: np.zeros((H, 6), dtype=np.float32) for k in PA}
    for nm, msk in [("pre_n", pre), ("pre_call", pre & is_call), ("pre_fold", pre & is_fold),
                    ("pre_aggr", pre & is_aggr), ("prew_n", pre & weak),
                    ("prew_call", pre & weak & is_call), ("prew_fold", pre & weak & is_fold),
                    ("post_n", post), ("post_check", post & is_check),
                    ("post_call", post & is_call), ("post_aggr", post & is_aggr),
                    ("post_fold", post & is_fold),
                    ("bestpass", post & (dec_best > 0) & (is_check | is_call)),
                    ("bestn", post & (dec_best > 0))]:
        TOT[nm] = hs(one, msk)

    # gemeinsam erreichte Streets je Sitzpaar
    reach = np.zeros((H, 6, 4), dtype=bool)
    reach[hloc, a_seat, a_str] = True

    # Aggression je Sitz UND Street (fuer gemeinsamen Druck in derselben Runde)
    s_aggr_st = np.zeros((H, 6, 4), dtype=np.float32)
    np.add.at(s_aggr_st, (hloc, a_seat, a_str), is_aggr.astype(np.float32))

    # Aggression gegen Dritte (Aggressor ist weder a noch b -> spaeter je Paar)
    s_aggr_when3 = hs(one, is_aggr & (a_pa >= 3))
    s_fold_when3 = hs(one, is_fold & faces & (a_pa >= 3))

    # ---- Paar-Hand-Merkmale ----
    names: list[str] = []
    feats: list[np.ndarray] = []

    def add(nm, arr):
        names.append(nm)
        feats.append(np.asarray(arr, dtype=np.float32))

    torder = np.arange(H, dtype=np.int32)   # chronologischer Rang (hsel ist zeitsortiert)
    blocks_key, blocks_h, blocks_t, blocks_X = [], [], [], []
    for (i, j) in COMBOS:
        names, feats = [], []
        a, b = sp[:, i], sp[:, j]
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b).astype(np.int64)
        # ordnungsinvariante Rollenzuweisung erfolgt ueber sym()/asym() unten
        ci, cj = contrib_bb[:, i], contrib_bb[:, j]
        ni, nj = net_bb[:, i], net_bb[:, j]
        wi, wj = wshare[:, i], wshare[:, j]
        fi, fj = folded[:, i].astype(np.float32), folded[:, j].astype(np.float32)
        sdi, sdj = showd[:, i].astype(np.float32), showd[:, j].astype(np.float32)

        # attribuierter Wertfluss (exakte Zerlegung: net = won_share*pot - contrib)
        flow_ij = ci * wj
        flow_ji = cj * wi

        def sym(x, y, nm):
            add(nm + "_sum", x + y)
            add(nm + "_absdiff", np.abs(x - y))
            add(nm + "_max", np.maximum(x, y))
            add(nm + "_min", np.minimum(x, y))

        sym(flow_ij, flow_ji, "flow")
        sym(ci, cj, "contrib")
        sym(ni, nj, "net")
        add("net_prod_neg", -(ni * nj))                     # gegenlaeufige Ergebnisse
        add("both_vpip", ((s_vpip[:, i] > 0) & (s_vpip[:, j] > 0)).astype(np.float32))
        add("both_showdown", sdi * sdj)
        add("one_folds_other_wins", np.maximum(fi * (wj > 0), fj * (wi > 0)).astype(np.float32))
        add("pot_bb", pot_bb)
        add("n_showdown", nsd)
        add("n_board", nboard)
        # heads-up: genau diese beiden ueberleben
        surv = (~folded).sum(1).astype(np.float32)
        add("headsup_pair", ((surv == 2) & (~folded[:, i]) & (~folded[:, j])).astype(np.float32))

        # gerichtete Konfrontation
        f_ij, f_ji = d_faced[:, i, j], d_faced[:, j, i]
        fo_ij, fo_ji = d_fold[:, i, j], d_fold[:, j, i]
        ca_ij, ca_ji = d_call[:, i, j], d_call[:, j, i]
        rr_ij, rr_ji = d_reraise[:, i, j], d_reraise[:, j, i]
        sym(f_ij, f_ji, "faced")
        sym(fo_ij, fo_ji, "foldto")
        sym(ca_ij, ca_ji, "callto")
        sym(rr_ij, rr_ji, "reraise")
        add("confront_any", ((f_ij + f_ji) > 0).astype(np.float32))
        # gefoldete Staerke gegen den Partner (Soft-Play-Signatur)
        sym(d_foldstr[:, i, j], d_foldstr[:, j, i], "foldstr")
        sym(d_callstr[:, i, j], d_callstr[:, j, i], "callstr")
        sym(d_callamt[:, i, j], d_callamt[:, j, i], "callamt")

        # gemeinsame Aggression gegen Dritte
        ag_i, ag_j = s_aggr[:, i], s_aggr[:, j]
        add("joint_aggr", np.minimum(ag_i, ag_j))
        add("joint_aggr3", np.minimum(s_aggr_when3[:, i], s_aggr_when3[:, j]))
        add("aggr_sum", ag_i + ag_j)
        add("mutual_aggr", np.minimum(rr_ij, 1) + np.minimum(rr_ji, 1))
        # Dritte, die in dieser Hand gefoldet haben
        third = [k for k in range(6) if k not in (i, j)]
        add("third_folds", folded[:, third].sum(1).astype(np.float32))
        add("third_contrib", contrib_bb[:, third].sum(1))
        add("third_net", net_bb[:, third].sum(1))
        add("pair_won_from_third", np.maximum(-(net_bb[:, third].sum(1)), 0.0) * ((wi + wj) > 0.5))

        # Karten / Entscheidungsqualitaet
        add("pre_i_j_max", np.maximum(pre_str[:, i], pre_str[:, j]))
        add("pre_i_j_min", np.minimum(pre_str[:, i], pre_str[:, j]))
        add("strong_fold_to_partner",
            np.maximum(np.where(fo_ij > 0, pre_str[:, i], 0.0), np.where(fo_ji > 0, pre_str[:, j], 0.0)))
        add("weak_call_to_partner",
            np.maximum(np.where(ca_ij > 0, 1 - pre_str[:, i], 0.0), np.where(ca_ji > 0, 1 - pre_str[:, j], 0.0)))
        # Verlierer hatte die staerkere Starthand
        loser_stronger = np.where(ni < nj, pre_str[:, i] - pre_str[:, j], pre_str[:, j] - pre_str[:, i])
        add("loser_pre_advantage", np.maximum(loser_stronger, 0.0))

        # ---- forensische Kartenmerkmale -------------------------------
        fold_ij = (fo_ij > 0).astype(np.float32)
        fold_ji = (fo_ji > 0).astype(np.float32)
        call_ij = (ca_ij > 0).astype(np.float32)
        call_ji = (ca_ji > 0).astype(np.float32)
        bi_, bj_ = is_best[:, i], is_best[:, j]
        si_, sj_ = str_fin[:, i], str_fin[:, j]
        # bester Blatt gegen den Partner weggeworfen
        sym(fold_ij * bi_, fold_ji * bj_, "foldbest_to")
        sym(fold_ij * (si_ > sj_), fold_ji * (sj_ > si_), "foldbetter_to")
        # dabei uebergebener Wert
        sym(fold_ij * bi_ * flow_ij, fold_ji * bj_ * flow_ji, "foldbest_value")
        # mit dem schlechteren Blatt gegen den Partner bezahlt
        sym(call_ij * (si_ < sj_) * ci, call_ji * (sj_ < si_) * cj, "callworse_paid")
        # Entscheidungsrang zum Aktionszeitpunkt (0 = bestes sichtbares Blatt)
        sym(fold_ij * (5 - n_better_dec[:, i]), fold_ji * (5 - n_better_dec[:, j]), "folddecrank")
        add("pair_str_gap", np.abs(n_better[:, i] - n_better[:, j]))
        add("pair_best_in_hand", np.maximum(bi_, bj_))
        # Gewinner hatte das schlechtere Blatt als der foldende Partner
        add("worse_hand_won", ((wi > 0.5) * (si_ < sj_) * fold_ji
                               + (wj > 0.5) * (sj_ < si_) * fold_ij).astype(np.float32))
        # Isolation: ein Dritter mit dem besten Blatt foldet, waehrend beide aggressiv sind
        third_best_fold = (is_best[:, third] * folded[:, third]).sum(1).astype(np.float32)
        add("third_best_folded", third_best_fold)
        add("iso_third_best_folded", third_best_fold * np.minimum(ag_i, ag_j))

        # ---- Partner-lebend-Kontrast (gepoolt ueber beide Mitglieder) ----
        for nm in PA:
            al = PA[nm][:, i, j] + PA[nm][:, j, i]
            de = (TOT[nm][:, i] - PA[nm][:, i, j]) + (TOT[nm][:, j] - PA[nm][:, j, i])
            add("pa_" + nm, al)
            add("pd_" + nm, de)
        wk_both = ((chen[:, i] <= 6.0) & (chen[:, j] <= 6.0)).astype(np.float32)
        st_both = ((chen[:, i] > 8.0) & (chen[:, j] > 8.0)).astype(np.float32)
        for nm in SURP:
            al = SURP[nm][:, i, j] + SURP[nm][:, j, i]
            de = (SURPT[nm][:, i] - SURP[nm][:, i, j]) + (SURPT[nm][:, j] - SURP[nm][:, j, i])
            add("surp_" + nm + "_alive", al)
            add("surp_" + nm + "_dead", de)
            add("surp_" + nm + "_diff", al - de)
            if ENGINE_WK and nm in ("s_all", "s_aggr"):
                # Ueberraschung genau dann, wenn KEIN Blatt die Handlung traegt
                add("surp_" + nm + "_wkboth", (al - de) * wk_both)
                add("surp_" + nm + "_stboth", (al - de) * st_both)
        sym(d_surpmax[:, i, j], d_surpmax[:, j, i], "surpmax_to")
        for nm in RES:
            al = RES[nm][:, i, j] + RES[nm][:, j, i]
            de = (RTOT[nm][:, i] - RES[nm][:, i, j]) + (RTOT[nm][:, j] - RES[nm][:, j, i])
            add("res_" + nm + "_alive", al)
            add("res_" + nm + "_dead", de)
            add("res_" + nm + "_diff", al - de)
        add("streets_together", (reach[:, i, :] & reach[:, j, :]).sum(1).astype(np.float32))

        # ---- Strukturmerkmale aus dem Evidenz/Nicht-Evidenz-Kontrast ----
        # Gemessen ueber alle 1860 gelabelten Paare: in 97 % der
        # coordinated_isolation-Evidenzhaende ist genau ein Partner aggressiv
        # und der andere foldet (Basis 48 %); die Aggression der Dritten faellt
        # dabei von 1.32 auf 0.92. `both_vpip` liegt in ALLEN Familien bei ~1.0
        # gegen eine Basis von 0.34-0.39.
        add("one_aggr_one_fold", (((ag_i > 0) & (fj > 0)) | ((ag_j > 0) & (fi > 0))).astype(np.float32))
        third_agg_tot = s_aggr[:, third].sum(1)
        add("third_aggr_total", third_agg_tot)
        add("third_aggr_suppressed", np.maximum(ag_i, ag_j) * (third_agg_tot == 0))
        add("pair_pot_share", (ci + cj) / np.maximum(pot_bb, 1e-6))
        add("pair_extract_from_third", (wi + wj) * pot_bb - (ci + cj))
        if ENGINE_WK:
            extr = (wi + wj) * pot_bb - (ci + cj)
            oaof = (((ag_i > 0) & (fj > 0)) | ((ag_j > 0) & (fi > 0))).astype(np.float32)
            for bn, bv in (("thirdaggr", third_agg_tot), ("extract", extr),
                           ("pot", pot_bb), ("isopat", oaof)):
                add(f"wk_{bn}_wkboth", bv * wk_both)
                add(f"wk_{bn}_stboth", bv * st_both)
        add("hu_streets_pair", (reach[:, i, :] & reach[:, j, :] &
                                ~reach[:, third, :].any(1)).sum(1).astype(np.float32))

        # ---- gemeinsame Praeflop-Eskalation mit Muellkarten --------------
        # Kernbefund fuer coordinated_isolation: beide Partner sind praeflop in
        # derselben Hand aggressiv, obwohl BEIDE Startkarten halten, die das
        # nicht rechtfertigen. Placebo-Kontrolle: dasselbe Aggregat mit starken
        # Karten trennt nicht (AUC 0.50 bei gleicher Basisrate) — das Signal
        # sitzt vollstaendig im Schwachkartenteil. Die erwartete Wirkung
        # (Opfer wird herausgedraengt) tritt dagegen NICHT ein: die Fold-Rate
        # Dritter ist unveraendert und das Paar verliert im Mittel Chips.
        api = (s_aggr_pre[:, i] > 0)
        apj = (s_aggr_pre[:, j] > 0)
        both_pf = (api & apj).astype(np.float32)
        # FEHLER BEHOBEN 2026-09-09: diese beiden Namen ueberschrieben bis hierher
        # `wi, wj = wshare[:, i], wshare[:, j]` (Z. 544). Die spaeteren
        # Verwendungen in `third_value_to_pair` (Z. 744) und `pair_winner_cat`
        # (Z. 777) meinen den GEWINNANTEIL, bekamen aber die Chen-Maske.
        # Folge: `third_value_to_pair` feuerte in 96 % statt 33 % der Zeilen,
        # und `pair_winner_cat` verglich zwei Wahrheitswerte statt zweier
        # Gewinnanteile und war damit nicht sitzinvariant.
        wkI = chen[:, i] <= 6.0
        wkJ = chen[:, j] <= 6.0
        add("both_aggr_pf", both_pf)
        add("ba_bothweak", both_pf * (wkI & wkJ))
        add("ba_bothstrong", both_pf * ((chen[:, i] > 8.0) & (chen[:, j] > 8.0)))  # Placebo
        add("ba_minchen", both_pf * np.minimum(chen[:, i], chen[:, j]))
        add("chen_min", np.minimum(chen[:, i], chen[:, j]))
        add("chen_max", np.maximum(chen[:, i], chen[:, j]))
        # Raise-Krieg: ein Partner erhoeht ueber die Aggression des anderen
        rw_pre = (d_aggr_st[0, :, i, j] + d_aggr_st[0, :, j, i])
        rw_all = (rr_ij + rr_ji)
        add("raise_over_partner_pf", rw_pre)
        add("raise_over_partner_any", rw_all)
        add("rw_bothweak", (rw_all > 0) * (wkI & wkJ))
        add("raisewar_iso_fold", (rw_all > 0) * ((fo_ij + fo_ji) > 0)
            * (folded[:, third].sum(1) > 0))
        # Entscheidungsdosis Dritter unter dem Druck des Paares
        tf = d_any[:, third, i].sum(1) + d_any[:, third, j].sum(1)
        tfo = d_fold[:, third, i].sum(1) + d_fold[:, third, j].sum(1)
        add("third_faces_pair_n", tf)
        add("third_fold_to_pair", tfo)
        add("third_foldrate_vs_pair", tfo / np.maximum(tf, 1.0))
        # Position des zuerst praeflop-aggressiven Partners (UTG = pos 3)
        fp_pos = np.where(api, pos[:, i].astype(np.float32), 9.0)
        fp_pos = np.where(apj, np.minimum(fp_pos, pos[:, j].astype(np.float32)), fp_pos)
        add("first_aggr_partner_pos", np.where(fp_pos < 9, fp_pos, -1.0))
        add("first_aggr_is_utg", ((fp_pos == 3) & (api | apj)).astype(np.float32))

        # ---- Zangenmuster (coordinated_isolation) -----------------------
        # Kern der Familie: ein Dritter geraet zwischen die Aggression BEIDER
        # Partner und gibt auf, waehrend die Partner den Konflikt untereinander
        # vermeiden. Das ist eine Struktur ueber drei Spieler und faellt in rein
        # paarweisen Merkmalen komplett heraus.
        fi3 = d_faced[:, third, i]                 # (H,4) Dritte gegen i
        fj3 = d_faced[:, third, j]
        sq = (fi3 > 0) & (fj3 > 0)                 # von beiden unter Druck gesetzt
        fold3 = (d_fold[:, third, i] + d_fold[:, third, j]) > 0
        best3 = is_best[:, third] > 0
        add("squeeze_count", sq.sum(1).astype(np.float32))
        add("squeeze_fold", (sq & fold3).sum(1).astype(np.float32))
        add("squeeze_fold_best", (sq & fold3 & best3).sum(1).astype(np.float32))
        add("squeeze_value", (sq * contrib_bb[:, third]).sum(1))
        add("both_aggr_street", np.minimum(s_aggr_st[:, i, :], s_aggr_st[:, j, :]).sum(1))
        add("aggr_street_overlap", ((s_aggr_st[:, i, :] > 0) & (s_aggr_st[:, j, :] > 0)).sum(1).astype(np.float32))
        # Eskalation gegeneinander, dann sofortiges Nachgeben, waehrend Dritte
        # foldeten. In den geplanten Evidenzhaenden dieser Familie ist das der
        # Kern: die beiden treiben den Pot GEGENEINANDER hoch, die Dritten
        # zahlen ein oder werden gebunden, danach gibt einer der beiden auf --
        # ein echter Showdown zwischen den Partnern kommt nie zustande.
        esc_yield = (((rr_ji > 0) & (fo_ij > 0)) | ((rr_ij > 0) & (fo_ji > 0))).astype(np.float32)
        add("iso_escalate_yield", esc_yield * (folded[:, third].sum(1) > 0))
        add("iso_esc_thirdmoney", esc_yield * contrib_bb[:, third].sum(1))
        add("iso_esc_pot", esc_yield * pot_bb)
        add("iso_esc_invested", esc_yield * np.minimum(ci, cj))
        # ruecksichtslose Aggression bei lebendem Partner (Potaufbau ohne Blatt)
        add("reckless_alive", RECK[:, i, j] + RECK[:, j, i])
        add("reckless_dead", (RECKT[:, i] - RECK[:, i, j]) + (RECKT[:, j] - RECK[:, j, i]))
        add("reckless_diff", (RECK[:, i, j] + RECK[:, j, i])
            - ((RECKT[:, i] - RECK[:, i, j]) + (RECKT[:, j] - RECK[:, j, i])))
        # von den Dritten eingesammelter Wert, wenn das Paar den Pot nimmt
        if ENGINE_CF:
            # --- gegenfaktischer Nutzen: eigener Verlust, Partnergewinn, Drittgewinn ---
            cf_pair = CFTO[:, i, j] + CFTO[:, j, i]
            cf_third = (CFTO[:, i, third].sum(1) + CFTO[:, j, third].sum(1))
            cf_self = CFSELF[:, i] + CFSELF[:, j]
            add("cf_self", cf_self)
            add("cf_topair", cf_pair)
            add("cf_tothird", cf_third)
            add("cf_diff", cf_pair - cf_third)
            add("cf_share", cf_pair / np.maximum(np.abs(cf_pair) + np.abs(cf_third), 1e-6))
            add("cf_effic", cf_pair / np.maximum(cf_self, 1e-4))
            add("cf_dirmax", np.maximum(CFTO[:, i, j], CFTO[:, j, i]))
            add("cf_dirmin", np.minimum(CFTO[:, i, j], CFTO[:, j, i]))
        add("third_value_to_pair", contrib_bb[:, third].sum(1) * ((wi + wj) > 0.5))
        # Die vor dem 2026-09-09 fehlerhafte Fassung dieser Zeile benutzte statt
        # der Gewinnanteile die Chen-Maske und mass damit versehentlich eine
        # ANDERE, offenbar nuetzliche Groesse: Geld der Dritten in Haenden, in
        # denen mindestens ein Partner ein schwaches Blatt haelt. Die Reparatur
        # allein kostete -0.0234 ap_universe (logs/fixtest.log). Die Groesse ist
        # regelkonform und mechanisch sinnvoll (coordinated_isolation: das Paar
        # eskaliert mit Muellkarten und laesst Dritte einzahlen), deshalb bleibt
        # sie erhalten — jetzt aber unter ihrem eigenen, ehrlichen Namen.
        add("third_money_pair_weak", contrib_bb[:, third].sum(1) * ((wkI + wkJ) > 0.5))

        # ---- Entscheidungsqualitaet gegen den Partner --------------------
        # Equity beim Fold/Call gegen den Partner — und, als interne Kontrolle,
        # dieselbe Groesse gegen ALLE uebrigen Gegner in denselben Haenden.
        for nm, dsum, dn, ssum, sn in (("eqf", d_eqf_sum, d_eqf_n, s_eqf_sum, s_eqf_n),
                                       ("eqc", d_eqc_sum, d_eqc_n, s_eqc_sum, s_eqc_n)):
            ap_ = dsum[:, i, j] + dsum[:, j, i]
            an_ = dn[:, i, j] + dn[:, j, i]
            add(nm + "_sum_p", ap_)
            add(nm + "_n_p", an_)
            add(nm + "_sum_o", (ssum[:, i] - dsum[:, i, j]) + (ssum[:, j] - dsum[:, j, i]))
            add(nm + "_n_o", (sn[:, i] - dn[:, i, j]) + (sn[:, j] - dn[:, j, i]))
        sym(d_eqf_max[:, i, j], d_eqf_max[:, j, i], "eqfold_max")
        sym(d_eqf_val[:, i, j], d_eqf_val[:, j, i], "eqfold_value")
        sym(d_qfoldbest[:, i, j], d_qfoldbest[:, j, i], "q_foldbest")
        sym(d_qfoldahead[:, i, j], d_qfoldahead[:, j, i], "h2h_foldahead")
        sym(d_qfoldahead_cheap[:, i, j], d_qfoldahead_cheap[:, j, i], "h2h_foldahead_cheap")
        sym(d_qfoldahead_cat[:, i, j], d_qfoldahead_cat[:, j, i], "h2h_foldahead_cat")
        sym(d_qcallbehind[:, i, j], d_qcallbehind[:, j, i], "h2h_callbehind")
        sym(d_qfoldcat[:, i, j], d_qfoldcat[:, j, i], "q_foldcat")
        sym(d_qfoldmade[:, i, j], d_qfoldmade[:, j, i], "q_foldmade")
        sym(d_qcallworst[:, i, j], d_qcallworst[:, j, i], "q_callworst")
        sym(d_lastfold[:, i, j], d_lastfold[:, j, i], "lastfold_to")
        sym(d_foldodds[:, i, j], d_foldodds[:, j, i], "foldodds")
        sym(d_foldtocall[:, i, j], d_foldtocall[:, j, i], "foldtocall")
        sym(d_foldinvested[:, i, j], d_foldinvested[:, j, i], "foldinvested")
        for st_, snm in enumerate(("pre", "flop", "turn", "riv")):
            sym(d_fold_st[st_, :, i, j], d_fold_st[st_, :, j, i], f"foldto_{snm}")
            sym(d_aggr_st[st_, :, i, j], d_aggr_st[st_, :, j, i], f"aggrto_{snm}")
        add("checkbest_pair", np.minimum(s_checkbest[:, i], s_checkbest[:, j]))
        add("checkbest_any", np.maximum(s_checkbest[:, i], s_checkbest[:, j]))
        # der Gewinner haette mit einer reinen High Card gewonnen (Bluff durchgekommen)
        winner_cat = np.where(wi > wj, cat_lev[3, np.arange(H), i], cat_lev[3, np.arange(H), j])
        # dieselbe Geschichte: die alte, fehlerhafte Fassung verglich die beiden
        # Chen-Masken. Als eigene Groesse ist das die Handkategorie desjenigen
        # Partners, der NICHT das schwaechere Startblatt hielt.
        weak_cat = np.where(wkI > wkJ, cat_lev[3, np.arange(H), i],
                            cat_lev[3, np.arange(H), j])
        add("pair_winner_cat", winner_cat.astype(np.float32))
        add("pair_weak_cat", weak_cat.astype(np.float32))

        # ---- Verhalten bedingt auf die VERDECKTEN Karten des Partners ----
        # Kernidee: ein unbeteiligter Spieler kann die Hole Cards des Gegenueber
        # nicht kennen. Seine Entscheidungen duerfen daher nicht mit deren
        # Staerke korrelieren, ausser ueber die sichtbaren Aktionen des Gegners.
        # pre[] ist ein Perzentil (Mittelwert 0.5), der Staerkerang ebenso;
        # der Mittelwert des Produkts ueber die gemeinsamen Haende ist damit
        # direkt eine Kovarianzschaetzung.
        prei_c, prej_c = pre_str[:, i] - 0.5, pre_str[:, j] - 0.5
        stri_c = (5.0 - n_better[:, i]) / 5.0 - 0.5
        strj_c = (5.0 - n_better[:, j]) / 5.0 - 0.5
        agi = (ag_i > 0).astype(np.float32)
        agj = (ag_j > 0).astype(np.float32)
        vpi = (s_vpip[:, i] > 0).astype(np.float32)
        vpj = (s_vpip[:, j] > 0).astype(np.float32)
        sym(fold_ij * prej_c, fold_ji * prei_c, "cov_fold_ppre")
        sym(call_ij * prej_c, call_ji * prei_c, "cov_call_ppre")
        sym(agi * prej_c, agj * prei_c, "cov_aggr_ppre")
        sym(vpi * prej_c, vpj * prei_c, "cov_vpip_ppre")
        sym(ci * prej_c, cj * prei_c, "cov_contrib_ppre")
        sym(fold_ij * strj_c, fold_ji * stri_c, "cov_fold_pfin")
        sym(agi * strj_c, agj * stri_c, "cov_aggr_pfin")
        # Kontrolle gegen die harmlose Erklaerung: nur Haende, in denen der
        # Partner NICHT aggressiv war, seine Staerke also unbeobachtbar blieb.
        pasj = (ag_j == 0).astype(np.float32)
        pasi = (ag_i == 0).astype(np.float32)
        sym(fold_ij * prej_c * pasj, fold_ji * prei_c * pasi, "cov_fold_ppre_pass")
        sym(vpi * prej_c * pasj, vpj * prei_c * pasi, "cov_vpip_ppre_pass")

        # Position / Stack
        add("stack_min", np.minimum(stack_bb[:, i], stack_bb[:, j]))
        add("pos_gap", np.abs(pos[:, i].astype(np.float32) - pos[:, j]))

        X = np.stack(feats, axis=1)
        blocks_key.append(lo * NP_PLAYERS + hi)
        blocks_h.append(hsel)
        blocks_t.append(torder)
        blocks_X.append(X)

    key = np.concatenate(blocks_key)
    hix = np.concatenate(blocks_h)
    tix = np.concatenate(blocks_t)
    X = np.concatenate(blocks_X, axis=0)
    return key, hix, tix, X, names

def action_bins_of(hsel: np.ndarray):
    """Situationsklasse und Aktionsklasse fuer alle Aktionen der Haende `hsel`.

    Getrennte, schlanke Implementierung fuer den einmaligen Aufbau der
    Erwartungstabelle (expect.py). Die Definitionen sind identisch zu
    table_pair_hands; Abweichungen wuerden die Bereinigung entwerten.
    """
    H = len(hsel)
    if H == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)
    btn = A("hand_button_seat")[hsel].astype(np.int64)
    board = A("hand_board")[hsel].astype(np.int16)
    h1 = A("seat_hole1")[hsel].astype(np.int16)
    h2 = A("seat_hole2")[hsel].astype(np.int16)
    pre_str = preflop_strength(h1, h2)
    holes = np.stack([h1, h2], axis=2)
    cat_lev = np.empty((4, H, 6), dtype=np.int8)
    for li, nc in enumerate((0, 3, 4, 5)):
        cc = holes if nc == 0 else np.concatenate(
            [holes, np.broadcast_to(board[:, None, :nc], (H, 6, nc))], axis=2)
        cat_lev[li] = (handeval.evaluate(cc.reshape(-1, cc.shape[2])).reshape(H, 6)
                       // (handeval.B ** 6)).astype(np.int8)
    off = A("act_offsets")
    lens = off[hsel + 1] - off[hsel]
    sel = np.concatenate([np.arange(off[h], off[h + 1]) for h in hsel])
    hloc = np.repeat(np.arange(H, dtype=np.int32), lens)
    a_seat = A("act_seat")[sel].astype(np.int64)
    a_act = A("act_action")[sel].astype(np.int8)
    a_str = A("act_street")[sel].astype(np.int64)
    a_amt = A("act_amount")[sel].astype(np.float32)
    a_pot = A("act_pot_before")[sel].astype(np.float32)
    a_tc = A("act_to_call")[sel].astype(np.float32)
    a_pa = A("act_players_active")[sel].astype(np.int64)
    dec_cat = cat_lev[a_str, hloc, a_seat]
    pos = (a_seat - btn[hloc]) % 6
    b = action_bin(a_str, a_tc, a_pot, a_pa, dec_cat, pre_str[hloc, a_seat], pos)
    is_aggr = (a_act == A_BET) | (a_act == A_RAISE) | ((a_act == A_ALLIN) & (a_amt > a_tc))
    cls = np.where(is_aggr, 0,
                   np.where(a_act == A_FOLD, 1,
                            np.where(a_act == A_CHECK, 3, 2))).astype(np.int8)
    return b, cls

POLICY_V2 = bool(int(os.environ.get("POLICY_V2", "0")))
POLICY_FIX = bool(int(os.environ.get("POLICY_FIX", "0")))
POLICY_Q2 = bool(int(os.environ.get("POLICY_Q2", "0")))
# Zweiter Q2-Stand (leckfreier Arm) bekommt ein eigenes Suffix, damit der
# V1LP-Bestand nicht ueberschrieben wird. Nur mit POLICY_Q2=1 wirksam.
POLICY_Q2_SFX = os.environ.get("POLICY_Q2_SFX", "_q2")
# SURP_STRAT: die Ueberraschung wird forensisch nach (Street, finaler
# Boardkartenzahl) zentriert. Das Aktionsmodell bleibt leckfrei; die
# Stratifizierung ist eine nachtraegliche Auswertung des vollstaendig
# beobachteten Handverlaufs, also dieselbe Klasse wie die Kartenforensik.
SURP_STRAT = bool(int(os.environ.get("SURP_STRAT", "0")))
ENGINE_WK = bool(int(os.environ.get("ENGINE_WK", "0")))
ENGINE_SZ = bool(int(os.environ.get("ENGINE_SZ", "0")))
ENGINE_CF = bool(int(os.environ.get("ENGINE_CF", "0")))
POLICY_EXTRA = [
    "board_paired", "board_flush3", "board_straighty", "board_hi",
    "raises_street", "callers_street", "eff_stack_bb", "draw_flush", "draw_oesd",
]
POLICY_FEATURES = [
    "street", "position", "players_active", "to_call_bb", "pot_bb", "pot_odds",
    "stack_bb", "to_call_over_stack", "dec_cat", "pre_pct", "n_board", "facing",
    "aggr_before_street",
] + (POLICY_EXTRA if POLICY_V2 else [])
_POLICY = None


def load_policy():
    """LightGBM-Policy: P(fold|check|call|aggressiv | Situation).

    Rein deskriptives Modell ueber alle 18,6 Mio Aktionen aller 12.000 Spieler.
    Es benutzt KEINE Labels und keine Spieleridentitaet, kann also kein
    Validierungslabel zurueckfuehren. Es liefert je Aktion die Ueberraschung
    -log P(beobachtete Aktion), also ein Mass dafuer, wie ungewoehnlich die
    Entscheidung in genau dieser Situation war.
    """
    global _POLICY
    if _POLICY is None:
        # Getrennte Dateien je Merkmalsstand: ein 22-Merkmal-Modell auf einer
        # 13-Merkmal-Matrix waere kein Fehler, den LightGBM meldet, sondern
        # stiller Unsinn.
        f = CACHE / ("policy_model_v2.txt" if POLICY_V2 else
                     ("policy_model_pf.txt" if POLICY_FIX else "policy_model.txt"))
        if f.exists():
            import lightgbm as lgb
            _POLICY = lgb.Booster(model_file=str(f))
            assert _POLICY.num_feature() == len(POLICY_FEATURES), (
                f"Policy-Modell hat {_POLICY.num_feature()} Merkmale, "
                f"die Matrix {len(POLICY_FEATURES)} — falsche Datei fuer POLICY_V2={POLICY_V2}")
        else:
            _POLICY = False
    return _POLICY or None


def _policy_matrix(a_str, a_pos, a_pa, a_tc, a_pot, a_stack, dec_cat,
                   pre_pct, n_board, aggr_before, a_bb):
    return np.stack([
        a_str.astype(np.float32), a_pos.astype(np.float32), a_pa.astype(np.float32),
        a_tc / a_bb, a_pot / a_bb, a_tc / np.maximum(a_pot + a_tc, 1.0),
        a_stack / a_bb, a_tc / np.maximum(a_stack, 1.0),
        dec_cat.astype(np.float32), pre_pct.astype(np.float32),
        n_board.astype(np.float32), (a_tc > 0).astype(np.float32),
        aggr_before.astype(np.float32),
    ], axis=1).astype(np.float32)



def _policy_extra(board, nb, h1, h2, a_str, hloc, a_seat, is_aggr, a_act,
                  a_stack, bb, ss):
    """Nine card-information features, vectorized using bit masks."""
    H = board.shape[0]
    NC = np.array([0, 3, 4, 5])                     # Boardkarten je Street
    br = np.where(board >= 0, board // 4, -1)
    bs = np.where(board >= 0, board % 4, -1)
    paired = np.zeros((4, H), np.float32); flush3 = np.zeros((4, H), np.float32)
    straighty = np.zeros((4, H), np.float32); hi = np.zeros((4, H), np.float32)
    bmask = np.zeros((4, H), np.int64)
    for li, nc in enumerate(NC):
        if nc == 0:
            continue
        r = br[:, :nc]
        m = np.bitwise_or.reduce(np.left_shift(1, r), axis=1)
        bmask[li] = m
        paired[li] = (np.bitwise_count(m) < nc).astype(np.float32)
        cnt = np.stack([(bs[:, :nc] == su).sum(1) for su in range(4)])
        flush3[li] = (cnt.max(0) >= 3).astype(np.float32)
        hi[li] = r.max(1).astype(np.float32)
        w = np.zeros(H, bool)
        for st in range(9):
            w |= np.bitwise_count(m & (0b11111 << st)) >= 3
        straighty[li] = w.astype(np.float32)
    # Zugmoeglichkeiten je (Hand, Sitz, Street) aus eigenen Karten + Board
    r1, r2 = h1 // 4, h2 // 4
    s1, s2 = h1 % 4, h2 % 4
    own = np.left_shift(1, r1) | np.left_shift(1, r2)
    dflush = np.zeros((4, H, 6), np.float32); doesd = np.zeros((4, H, 6), np.float32)
    for li, nc in enumerate(NC):
        cm = own | bmask[li][:, None]
        w = np.zeros((H, 6), bool)
        for st in range(10):
            w |= np.bitwise_count(cm & (0b1111 << st)) == 4
        doesd[li] = w.astype(np.float32)
        for su in range(4):
            c = ((s1 == su).astype(np.int8) + (s2 == su).astype(np.int8))
            if nc:
                c = c + (bs[:, :nc] == su).sum(1)[:, None]
            dflush[li] = np.maximum(dflush[li], (c == 4).astype(np.float32))
    # Zaehler VOR der jeweiligen Aktion, gruppiert nach (Hand, Street)
    grp = hloc.astype(np.int64) * 4 + a_str
    newg = np.r_[True, grp[1:] != grp[:-1]]
    starts = np.flatnonzero(newg)
    gid = np.cumsum(newg) - 1

    def before(flag):
        pre = np.r_[0, np.cumsum(flag)[:-1]]
        return (pre - pre[starts][gid]).astype(np.float32)

    raises_street = before(is_aggr.astype(np.int64))
    callers_street = before((a_act == A_CALL).astype(np.int64))
    # Effektiver Stack: der eigene aktuelle Stack gegen den groessten STARTstack
    # der uebrigen Sitze. Nur damit ist "wieviel kann hier ueberhaupt noch
    # verloren gehen" abgebildet; `stack_bb` allein sagt das nicht.
    srt = np.sort(ss, axis=1)
    top1, top2 = srt[:, -1], srt[:, -2]
    own_start = ss[np.arange(H)[:, None], np.arange(6)[None, :]]
    other_max = np.where(own_start >= top1[:, None], top2[:, None], top1[:, None])
    eff = np.minimum(a_stack, other_max[hloc, a_seat]) / np.maximum(bb[hloc], 1e-6)
    return (paired[a_str, hloc], flush3[a_str, hloc], straighty[a_str, hloc],
            hi[a_str, hloc], raises_street, callers_street, eff.astype(np.float32),
            dflush[a_str, hloc, a_seat], doesd[a_str, hloc, a_seat])


def _aggr_before_street(hloc, a_str, is_aggr):
    """Zahl der aggressiven Aktionen vorher in derselben (Hand, Street)."""
    n = len(hloc)
    c = np.concatenate([[0], np.cumsum(is_aggr.astype(np.int64))])[:n]
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = (hloc[1:] != hloc[:-1]) | (a_str[1:] != a_str[:-1])
    base = np.maximum.accumulate(np.where(new, c, 0))
    return (c - base).astype(np.float32)


def policy_features(hsel: np.ndarray):
    """Merkmalsmatrix und Aktionsklasse fuer alle Aktionen der Haende `hsel`."""
    H = len(hsel)
    if H == 0:
        return np.zeros((0, len(POLICY_FEATURES)), np.float32), np.zeros(0, np.int8)
    btn = A("hand_button_seat")[hsel].astype(np.int64)
    bb = A("hand_bb")[hsel].astype(np.float32)
    nb = A("hand_n_board")[hsel].astype(np.float32)
    board = A("hand_board")[hsel].astype(np.int16)
    h1 = A("seat_hole1")[hsel].astype(np.int16)
    h2 = A("seat_hole2")[hsel].astype(np.int16)
    pre_pct = preflop_strength(h1, h2)
    holes = np.stack([h1, h2], axis=2)
    str_lev = np.empty((4, H, 6), dtype=np.int64)
    for li, nc in enumerate((0, 3, 4, 5)):
        cc = holes if nc == 0 else np.concatenate(
            [holes, np.broadcast_to(board[:, None, :nc], (H, 6, nc))], axis=2)
        str_lev[li] = handeval.evaluate(cc.reshape(-1, cc.shape[2])).reshape(H, 6)
    cat_lev = (str_lev // (handeval.B ** 6)).astype(np.int8)
    nb_lev = np.stack([(str_lev[l][:, None, :] > str_lev[l][:, :, None]).sum(2)
                       for l in range(4)]).astype(np.int8)
    off = A("act_offsets")
    lens = off[hsel + 1] - off[hsel]
    sel = np.concatenate([np.arange(off[h], off[h + 1]) for h in hsel])
    hloc = np.repeat(np.arange(H, dtype=np.int32), lens)
    a_seat = A("act_seat")[sel].astype(np.int64)
    a_act = A("act_action")[sel].astype(np.int8)
    a_str = A("act_street")[sel].astype(np.int64)
    a_amt = A("act_amount")[sel].astype(np.float32)
    a_amtto = A("act_amount_to")[sel].astype(np.float32)
    a_pot = A("act_pot_before")[sel].astype(np.float32)
    a_stack = A("act_stack_before")[sel].astype(np.float32)
    a_tc = A("act_to_call")[sel].astype(np.float32)
    a_pa = A("act_players_active")[sel].astype(np.int64)
    is_aggr = (a_act == A_BET) | (a_act == A_RAISE) | ((a_act == A_ALLIN) & (a_amt > a_tc))
    ano = np.concatenate([np.arange(l) for l in lens])
    a_frac = ano / np.maximum(lens[hloc] - 1, 1)
    nb_feat = (np.array([0, 3, 4, 5], np.float32)[a_str] if POLICY_FIX else nb[hloc])
    X = _policy_matrix(a_str, (a_seat - btn[hloc]) % 6, a_pa, a_tc, a_pot, a_stack,
                       cat_lev[a_str, hloc, a_seat], pre_pct[hloc, a_seat], nb_feat,
                       _aggr_before_street(hloc, a_str, is_aggr), bb[hloc])
    if POLICY_V2:
        extra = _policy_extra(board, nb, h1, h2, a_str, hloc, a_seat, is_aggr, a_act,
                              a_stack, bb, A("seat_stack")[hsel].astype(np.float32))
        X = np.concatenate([X, np.stack(extra, axis=1).astype(np.float32)], axis=1)
    cls = np.where(is_aggr, 0, np.where(a_act == A_FOLD, 1,
                                        np.where(a_act == A_CHECK, 3, 2))).astype(np.int8)
    return X, cls
