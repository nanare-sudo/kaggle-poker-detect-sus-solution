from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

from raw_cache import build
from validation import table_folds
from tdlib import _prefix
from current_finish import align


def write_raw_fixture(path, shuffle=False, duplicate_action=False):
    path.mkdir()
    players = pd.DataFrame(dict(player_id=[f'P{i}' for i in range(6)],
        account_age_days=20, experience_hands_bucket='low', preferred_stake='small',
        region_bucket='region', client_family='client'))
    hands = pd.DataFrame(dict(hand_id=['Hlater', 'Hearlier'], table_id=['T0', 'T0'],
        started_at=[20, 10], phase=['evaluation', 'development'], button_seat=[0, 1],
        small_blind=1, big_blind=2, final_pot=6, players_at_showdown=0,
        board_cards=['', '']))
    seats = pd.DataFrame([dict(hand_id=h, seat_no=i, player_id=f'P{i}',
        starting_stack=100, total_contribution=i % 2, net_chips=0, folded=True,
        went_to_showdown=False, won_share=0., hole_card_1='2c', hole_card_2='3d')
        for h in hands.hand_id for i in range(6)])
    actions = pd.DataFrame([dict(hand_id=h, action_no=i, player_id=f'P{i}',
        street='preflop', action='fold', amount=0, amount_to=0, pot_before=3,
        stack_before=100, to_call=2, players_active=6-i)
        for h in hands.hand_id for i in range(6)])
    if duplicate_action:
        actions.loc[1, 'action_no'] = 0
    for name, frame in [('players', players), ('hands', hands), ('seats', seats), ('actions', actions)]:
        # Retain the hand-index schema; permute the joined tables independently.
        if shuffle and name != 'hands':
            frame = frame.sample(frac=1, random_state=43)
        frame.to_parquet(path / (name + '.parquet'), index=False)
    pair = dict(pair_id=['pair'], player_1=['P0'], player_2=['P1'])
    pd.DataFrame(dict(**pair, label=[1], behavior_family=['soft_play'])).to_csv(
        path / 'development_labels.csv', index=False)
    pd.DataFrame(pair).to_csv(path / 'evaluation_pairs.csv', index=False)
    pd.DataFrame(dict(pair_id=['pair'], hand_id=['Hearlier'], evidence_rank=[1])).to_csv(
        path / 'development_evidence.csv', index=False)


def test_raw_joins_ignore_seat_and_action_row_order(tmp_path):
    write_raw_fixture(tmp_path / 'ordered')
    write_raw_fixture(tmp_path / 'shuffled', shuffle=True)
    build(tmp_path / 'ordered', tmp_path / 'cache1')
    build(tmp_path / 'shuffled', tmp_path / 'cache2')
    for p in (tmp_path / 'cache1').glob('*.npy'):
        np.testing.assert_array_equal(np.load(p, allow_pickle=True),
            np.load(tmp_path / 'cache2' / p.name, allow_pickle=True), err_msg=p.name)
    np.testing.assert_array_equal(np.load(tmp_path / 'cache2/act_seat.npy'), np.tile(np.arange(6), 2))
    np.testing.assert_array_equal(np.load(tmp_path / 'cache2/act_offsets.npy'), [0, 6, 12])


def test_duplicate_action_key_is_rejected(tmp_path):
    write_raw_fixture(tmp_path / 'raw', duplicate_action=True)
    with pytest.raises(AssertionError):
        build(tmp_path / 'raw', tmp_path / 'cache')


def test_table_folds_are_stable_under_subset_and_permutation():
    tables = np.arange(400)
    chosen = np.array([99, 4, 300, 0, 4, 122])
    for seed in (0, 1):
        np.testing.assert_array_equal(table_folds(chosen, seed=seed), table_folds(tables, seed=seed)[chosen])


def test_key_alignment_rejects_missing_and_duplicate_keys():
    np.testing.assert_array_equal(align(np.array([40, 10, 30]), np.array([30, 40])), [2, 0])
    with pytest.raises(AssertionError):
        align(np.array([1, 1]), np.array([1]))
    with pytest.raises(AssertionError):
        align(np.array([1, 2]), np.array([3]))


def test_capped_decoder_matches_enumeration():
    probabilities = np.array([0.15, 0.8, 0.3, 0.9])
    cap = 2
    got = _prefix(probabilities, cap)
    for n in range(len(probabilities) + 1):
        expected = np.zeros(cap + 1)
        for outcome in product((0, 1), repeat=n):
            probability = np.prod([p if y else 1-p for p, y in zip(probabilities, outcome)])
            expected[min(sum(outcome), cap)] += probability
        np.testing.assert_allclose(got[n], expected, atol=1e-15)
