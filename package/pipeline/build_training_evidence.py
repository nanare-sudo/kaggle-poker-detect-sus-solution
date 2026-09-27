"""Rebuild original final evidence training matrices from gameplay and labels.

Labels select training pairs, their family and target hands only. Within-pair
normalization and eight action records are computed by the serving functions.
No historical features, fitted model, routing or submission is read.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['SEQ_FEAT'] = '1'
os.environ['POLICY_Q2'] = '1'
os.environ['POLICY_Q2_SFX'] = '_q2'
for n in ['POLICY_FIX', 'POLICY_V2', 'SURP_STRAT', 'ENGINE_WK', 'ENGINE_SZ',
          'ENGINE_CF', 'AGG_QREPLACE', 'AGG_EPI', 'Q2_POOL', 'Q2_WIN_HEAD']:
    os.environ[n] = '0'
os.environ['AGG_QFRACS'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

STATE = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init(root):
    global STATE
    root = Path(root)
    sys.path.insert(0, str(root / 'src'))
    os.environ['POKER_CACHE'] = str(root / 'cache')
    os.environ['Q2_STATS_FILE'] = str(root / 'cache/q2_pstats.npz')
    os.environ['POKER_CHEN_ORDER_FILE'] = str(root / 'pipeline/chen_order_cpu.json')
    import pandas as pd
    lab = pd.read_parquet(root / 'cache/labels.parquet')
    evidence = pd.read_parquet(root / 'cache/evidence.parquet')
    key = lab.plo.to_numpy().astype(np.int64) * 12000 + lab.phi.to_numpy()
    positive = lab.label.to_numpy() == 1
    pair_to_key = dict(zip(lab.pair_id, key))
    player_table = np.full(12000, -1, np.int16)
    sp = np.load(root / 'cache/seat_player.npy', mmap_mode='r')
    ht = np.load(root / 'cache/hand_table.npy', mmap_mode='r')
    for seat in range(6):
        player_table[np.asarray(sp[:, seat])] = np.asarray(ht)
    families = dict(zip(key[positive], lab.behavior_family.to_numpy()[positive]))
    STATE = dict(keys=key[positive], table=player_table[key[positive] // 12000], families=families,
        evidence=set((pair_to_key[p], int(h)) for p, h in zip(evidence.pair_id, evidence.hidx)))


def work(table):
    started = time.time()
    want = STATE['keys'][STATE['table'] == table]
    if not len(want):
        return table, dict(empty=True), {}
    import engine as E
    import hand_features as S
    E.POLICY_Q2 = True
    E.POLICY_Q2_SFX = '_q2'
    k, h, to, base, names, role, rn, ci, cn, events, en = S.live_data(table, 0, want)
    np.testing.assert_array_equal(np.unique(k), np.sort(want))
    y = np.array([(int(q), int(hh)) in STATE['evidence'] for q, hh in zip(k, h)], np.int8)
    family = np.array([STATE['families'][int(q)] for q in k])
    data, schemas = dict(empty=False), {}
    for g, name in enumerate(S.G.FAMS):
        take = family == name
        if not take.any():
            continue
        matrix, columns = S.G.matrix(k[take], to[take], base[take], names,
            role[take], rn, ci[take], cn, events[take], en, g, 'CHRONO_ADD')
        assert np.isfinite(matrix).all()
        for n, values in [('key', k[take]), ('h', h[take]), ('torder', to[take]),
                          ('y', y[take]), ('X', matrix)]:
            data[f'g{g}_{n}'] = values
        schemas[str(g)] = columns
    print('EVIDENCE TRAINING FEATURES', table, len(want), len(k), round(time.time() - started), flush=True)
    return table, data, schemas


def main(a):
    root = a.root.resolve()
    sources = {str(p.relative_to(root)): sha(p) for p in (root / 'src').glob('*.py')}
    sources.update(driver=sha(Path(__file__)), chen=sha(root / 'pipeline/chen_order_cpu.json'))
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / 'specification.json'
    if path.exists():
        assert json.loads(path.read_text())['source_sha256'] == fingerprint
    else:
        path.write_text(json.dumps(dict(source_sha256=fingerprint, sources=sources,
            base_policy='_q2', event_policy='_q2nnnoply', arm='CHRONO_ADD',
            reads_historical_features=False, public_upload=False), indent=2))
    todo = []
    for table in range(a.tables):
        path = a.out / f'{table:03d}.npz'
        if path.exists():
            with np.load(path) as z:
                assert str(z['source_sha256']) == fingerprint
        else:
            todo.append(table)
    names_path = a.out / 'feature_names.json'
    names = json.loads(names_path.read_text()) if names_path.exists() else {}
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.jobs, initializer=init, initargs=(str(root),)) as pool:
        for table, data, schemas in pool.imap(work, todo):
            for g, columns in schemas.items():
                if g in names:
                    assert names[g] == columns
                names[g] = columns
            names_path.write_text(json.dumps(names))
            np.savez_compressed(a.out / f'{table:03d}.npz', **data, source_sha256=fingerprint)
    print('REQUESTED EVIDENCE TRAINING INPUTS COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tables', type=int, default=400)
    parser.add_argument('--jobs', type=int, default=2)
    main(parser.parse_args())
