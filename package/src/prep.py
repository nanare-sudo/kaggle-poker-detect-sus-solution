"""Kompakte, integer-kodierte Repraesentation der Wettbewerbsdaten.

Alle Tabellen werden einmalig in .npy-Dateien unter cache/ ueberfuehrt. Die
Zeilenreihenfolge der Parquet-Dateien wird nur als Performance-Abkuerzung
benutzt und dabei explizit VERIFIZIERT; schlaegt die Pruefung fehl, wird ueber
hand_id/player_id regulaer gejoint. Die Reihenfolge ist damit kein Merkmal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("POKER_DATA", str(ROOT / "data")))
CACHE = Path(os.environ.get("POKER_CACHE", str(ROOT / "cache")))

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_OF = {c: i for i, c in enumerate(RANKS)}
SUIT_OF = {c: i for i, c in enumerate(SUITS)}

STREETS = ["preflop", "flop", "turn", "river"]
ACTIONS = ["fold", "check", "call", "bet", "raise", "all_in", "post_blind", "post_ante"]


def encode_cards(series: pd.Series) -> np.ndarray:
    """'Ah' -> 0..51 (rank*4+suit), leer/NaN -> -1."""
    s = series.fillna("").astype(str).str.strip()
    out = np.full(len(s), -1, dtype=np.int8)
    ok = s.str.len() >= 2
    r = s[ok].str[0].map(RANK_OF)
    u = s[ok].str[1].map(SUIT_OF)
    good = r.notna() & u.notna()
    idx = np.flatnonzero(ok.to_numpy())[good.to_numpy()]
    out[idx] = (r[good].to_numpy() * 4 + u[good].to_numpy()).astype(np.int8)
    return out


def save(name: str, arr: np.ndarray) -> None:
    np.save(CACHE / f"{name}.npy", arr)


def build() -> None:
    CACHE.mkdir(exist_ok=True)

    # ---------------- players ----------------
    players = pd.read_parquet(DATA / "players.parquet")
    players = players.sort_values("player_id").reset_index(drop=True)
    pid = players["player_id"].to_numpy()
    pcode = {p: i for i, p in enumerate(pid)}
    np.save(CACHE / "player_ids.npy", pid)
    save("player_account_age", players["account_age_days"].to_numpy(np.int32))
    cat_maps = {}
    for col in ["experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"]:
        c = players[col].astype("category")
        cat_maps[col] = list(c.cat.categories)
        save(f"player_{col}", c.cat.codes.to_numpy(np.int8))

    # ---------------- hands ----------------
    hands = pd.read_parquet(DATA / "hands.parquet")
    hid = hands["hand_id"].to_numpy()
    n_hands = len(hands)
    np.save(CACHE / "hand_ids.npy", hid)
    tab = hands["table_id"].astype("category")
    cat_maps["table_id"] = list(tab.cat.categories)
    save("hand_table", tab.cat.codes.to_numpy(np.int16))
    save("hand_started_at", hands["started_at"].astype("int64").to_numpy())
    save("hand_phase", (hands["phase"].to_numpy() == "evaluation").astype(np.int8))
    save("hand_button_seat", hands["button_seat"].to_numpy(np.int8))
    save("hand_sb", hands["small_blind"].to_numpy(np.int16))
    save("hand_bb", hands["big_blind"].to_numpy(np.int16))
    save("hand_final_pot", hands["final_pot"].to_numpy(np.int32))
    save("hand_n_showdown", hands["players_at_showdown"].to_numpy(np.int8))

    board = hands["board_cards"].fillna("").astype(str).str.split()
    nb = board.str.len().fillna(0).to_numpy().astype(np.int8)
    save("hand_n_board", nb)
    bcards = np.full((n_hands, 5), -1, dtype=np.int8)
    exploded = board.explode()
    pos = exploded.groupby(level=0).cumcount()
    valid = exploded.notna()
    codes = encode_cards(exploded[valid])
    bcards[exploded.index[valid].to_numpy(), pos[valid].to_numpy()] = codes
    save("hand_board", bcards)
    del board, exploded, hands

    # ---------------- seats ----------------
    seats = pq.read_table(DATA / "seats.parquet").to_pandas()
    assert len(seats) == 6 * n_hands, "seats: nicht exakt 6 Zeilen je Hand"
    aligned = bool((seats["hand_id"].to_numpy()[::6] == hid).all()) and bool(
        (seats["seat_no"].to_numpy().reshape(-1, 6) == np.arange(6)).all()
    )
    if not aligned:  # Fallback: expliziter Join, Reihenfolge irrelevant
        hmap = pd.Series(np.arange(n_hands, dtype=np.int64), index=hid)
        seats["_h"] = seats["hand_id"].map(hmap).to_numpy()
        seats = seats.sort_values(["_h", "seat_no"], kind="stable").reset_index(drop=True)
        assert (seats["seat_no"].to_numpy().reshape(-1, 6) == np.arange(6)).all()
    print(f"seats row-order aligned: {aligned}")

    def col6(name, dtype):
        return seats[name].to_numpy(dtype).reshape(n_hands, 6)

    save("seat_player", seats["player_id"].map(pcode).to_numpy(np.int16).reshape(n_hands, 6))
    save("seat_stack", col6("starting_stack", np.int32))
    save("seat_contrib", col6("total_contribution", np.int32))
    save("seat_net", col6("net_chips", np.int32))
    save("seat_folded", col6("folded", np.bool_))
    save("seat_showdown", col6("went_to_showdown", np.bool_))
    save("seat_won_share", col6("won_share", np.float32))
    save("seat_hole1", encode_cards(seats["hole_card_1"]).reshape(n_hands, 6))
    save("seat_hole2", encode_cards(seats["hole_card_2"]).reshape(n_hands, 6))
    seat_player = np.load(CACHE / "seat_player.npy")
    del seats

    # ---------------- actions ----------------
    acts = pq.read_table(DATA / "actions.parquet").to_pandas()
    an = acts["action_no"].to_numpy()
    starts = np.flatnonzero(an == 0)
    aligned_a = bool(len(starts) == n_hands and (acts["hand_id"].to_numpy()[starts] == hid).all())
    if not aligned_a:
        hmap = pd.Series(np.arange(n_hands, dtype=np.int64), index=hid)
        acts["_h"] = acts["hand_id"].map(hmap).to_numpy()
        acts = acts.sort_values(["_h", "action_no"], kind="stable").reset_index(drop=True)
        starts = np.flatnonzero(acts["action_no"].to_numpy() == 0)
        assert len(starts) == n_hands
    if aligned_a:
        # Der schnelle Pfad vertraut der Dateizeilenreihenfolge. Er ist nur
        # zulaessig, wenn action_no innerhalb jeder Hand exakt 0..n-1 laeuft;
        # sonst wuerde die Aktionsreihenfolge aus der Datei statt aus dem
        # ausgewiesenen Feld kommen. Die Pruefung stand bisher nicht da.
        _off = np.append(starts, len(an)).astype(np.int64)
        _exp = np.arange(len(an), dtype=np.int64) - np.repeat(starts, np.diff(_off))
        assert np.array_equal(an.astype(np.int64), _exp), \
            "action_no laeuft nicht handweise 0..n-1 — schneller Pfad unzulaessig"
    print(f"actions row-order aligned: {aligned_a}")

    offsets = np.append(starts, len(acts)).astype(np.int64)
    save("act_offsets", offsets)
    a_hidx = np.repeat(np.arange(n_hands, dtype=np.int32), np.diff(offsets))
    save("act_hidx", a_hidx)

    st = acts["street"].astype(pd.CategoricalDtype(STREETS))
    assert st.isna().sum() == 0, f"unbekannte street: {set(acts['street'].unique()) - set(STREETS)}"
    save("act_street", st.cat.codes.to_numpy(np.int8))
    ac = acts["action"].astype(pd.CategoricalDtype(ACTIONS))
    unknown = set(acts["action"].unique()) - set(ACTIONS)
    assert not unknown, f"unbekannte action: {unknown}"
    save("act_action", ac.cat.codes.to_numpy(np.int8))
    save("act_amount", acts["amount"].to_numpy(np.int32))
    save("act_amount_to", acts["amount_to"].to_numpy(np.int32))
    save("act_pot_before", acts["pot_before"].to_numpy(np.int32))
    save("act_stack_before", acts["stack_before"].to_numpy(np.int32))
    save("act_to_call", acts["to_call"].to_numpy(np.int32))
    save("act_players_active", acts["players_active"].to_numpy(np.int8))

    # Akteur -> Sitzindex (0..5) innerhalb der Hand
    a_player = acts["player_id"].map(pcode).to_numpy(np.int16)
    save("act_player", a_player)
    seat_of = np.full(len(acts), -1, dtype=np.int8)
    CH = 2_000_000
    for lo in range(0, len(acts), CH):
        hi = min(lo + CH, len(acts))
        blk = seat_player[a_hidx[lo:hi]]                     # (n,6)
        eq = blk == a_player[lo:hi, None]
        seat_of[lo:hi] = np.where(eq.any(1), eq.argmax(1), -1).astype(np.int8)
    assert (seat_of >= 0).all(), "Aktion ohne passenden Sitz"
    save("act_seat", seat_of)
    del acts

    # ---------------- labels / pairs ----------------
    lab = pd.read_csv(DATA / "development_labels.csv")
    evi = pd.read_csv(DATA / "development_evidence.csv")
    evp = pd.read_csv(DATA / "evaluation_pairs.csv")
    hmap = pd.Series(np.arange(n_hands, dtype=np.int64), index=hid)
    lab["p1"] = lab["player_1"].map(pcode); lab["p2"] = lab["player_2"].map(pcode)
    evp["p1"] = evp["player_1"].map(pcode); evp["p2"] = evp["player_2"].map(pcode)
    lab["plo"] = np.minimum(lab.p1, lab.p2); lab["phi"] = np.maximum(lab.p1, lab.p2)
    evp["plo"] = np.minimum(evp.p1, evp.p2); evp["phi"] = np.maximum(evp.p1, evp.p2)
    evi["hidx"] = evi["hand_id"].map(hmap)
    lab.to_parquet(CACHE / "labels.parquet")
    evi.to_parquet(CACHE / "evidence.parquet")
    evp.to_parquet(CACHE / "eval_pairs.parquet")

    (CACHE / "cat_maps.json").write_text(json.dumps(cat_maps, indent=1))
    (CACHE / "meta.json").write_text(json.dumps({
        "n_hands": int(n_hands), "n_players": int(len(pid)),
        "n_actions": int(len(a_hidx)), "seats_aligned": aligned, "actions_aligned": aligned_a,
        "streets": STREETS, "actions_vocab": ACTIONS,
    }, indent=1))
    print("prep fertig ->", CACHE)
