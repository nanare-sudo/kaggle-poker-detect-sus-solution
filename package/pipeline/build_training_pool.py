"""Build historical pool-training aggregates from raw hands and fresh teachers."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['SEQ_FEAT'] = '1'
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


def init(root, teachers):
    global STATE
    root, teachers = Path(root), Path(teachers)
    sys.path.insert(0, str(root / 'src'))
    os.environ['POKER_CACHE'] = str(root / 'cache')
    os.environ['Q2_STATS_FILE'] = str(root / 'cache/q2_pstats.npz')
    os.environ['POKER_CHEN_ORDER_FILE'] = str(root / 'pipeline/chen_order_cpu.json')
    import lightgbm as lgb
    report = json.loads((teachers / 'training.json').read_text())
    assert report['status'] == 'PASS'
    models = [lgb.Booster(model_file=str(teachers / f'inst_M1_fs0_f{f}.txt')) for f in range(5)]
    STATE = dict(models=models, report=report)


def work(table):
    import engine as E
    import evidence as EV
    import seqwire
    from pool_training import _agg, AGG_NAMES
    started = time.time()
    E.POLICY_Q2 = False
    ht, hp, st = [E.A(n) for n in ['hand_table', 'hand_phase', 'hand_started_at']]
    hands = np.flatnonzero((ht == table) & (hp == 0))
    hands = hands[np.argsort(st[hands], kind='stable')]
    assert len(hands) == 3000
    k, h, to, X, names = E.table_pair_hands(hands)
    keys, counts = np.unique(k, return_counts=True)
    want = keys[counts >= 57]
    take = np.isin(k, want)
    k, h, to, X = k[take], h[take], to[take], X[take]
    order = np.lexsort((to, k))
    k, h, to, X = k[order], h[order], to[order], X[order]
    keep = EV.evid_keep(names)
    sequence, sequence_names = seqwire.live(hands, k, h)
    matrix = np.concatenate([X[:, keep], sequence], axis=1)
    model_names = [n for n, ok in zip(names, keep) if ok] + sequence_names
    fold = int(STATE['report']['table_fold'][table])
    model = STATE['models'][fold]
    assert model.feature_name() == model_names
    scores = model.predict(matrix, num_threads=1)
    keys, start, counts = np.unique(k, return_index=True, return_counts=True)
    positions = to.astype(np.float32) / max(len(hands) - 1, 1)
    q = STATE['report']['quantiles'][str(fold)]
    values = _agg(keys, start, counts, scores, positions, q['0.9'], q['0.99'])
    assert np.isfinite(values).all()
    print('POOL TRAINING FEATURES', table, len(keys), len(k), round(time.time() - started), flush=True)
    return table, dict(key=keys, X=values, table=np.full(len(keys), table, np.int16),
                      fold=fold, hand_rows=len(k)), AGG_NAMES


def main(a):
    root, teachers = a.root.resolve(), a.teachers.resolve()
    receipt = json.loads((teachers / 'training.json').read_text())
    assert receipt['status'] == 'PASS'
    sources = {str(p.relative_to(root)): sha(p) for p in (root / 'src').glob('*.py')}
    sources.update(driver=sha(Path(__file__)), chen=sha(root / 'pipeline/chen_order_cpu.json'),
                   teacher_receipt=sha(teachers / 'training.json'))
    for f in range(5):
        sources[f'teacher_{f}'] = sha(teachers / f'inst_M1_fs0_f{f}.txt')
        assert sources[f'teacher_{f}'] == receipt['models'][str(f)]['sha256']
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = a.out / 'specification.json'
    if spec.exists():
        assert json.loads(spec.read_text())['source_sha256'] == fingerprint
    else:
        spec.write_text(json.dumps(dict(source_sha256=fingerprint, sources=sources,
            minimum_shared_hands=57, reads_historical_features=False, public_upload=False), indent=2))
    todo = []
    for table in range(a.tables):
        path = a.out / f'{table:03d}.npz'
        if path.exists():
            with np.load(path) as z:
                assert str(z['source_sha256']) == fingerprint
        else:
            todo.append(table)
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.jobs, initializer=init,
                                  initargs=(str(root), str(teachers))) as pool:
        for table, values, names in pool.imap(work, todo):
            path = a.out / 'feature_names.json'
            if path.exists():
                assert json.loads(path.read_text()) == names
            else:
                path.write_text(json.dumps(names))
            np.savez_compressed(a.out / f'{table:03d}.npz', **values, source_sha256=fingerprint)
    print('REQUESTED POOL TRAINING INPUTS COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tables', type=int, default=400)
    parser.add_argument('--jobs', type=int, default=2)
    main(parser.parse_args())
