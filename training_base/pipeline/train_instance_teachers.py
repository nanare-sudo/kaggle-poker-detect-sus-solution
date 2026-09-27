"""Reproduce the five original M1 fs0 pool teachers from rebuilt hand inputs.

The original models used a seeded partition of the 397 tables with labeled
pairs. This historical partition is explicit here for asset reproduction only;
modern validation must keep its fixed 400-table assignment. IDs group folds,
never enter model features. No retained model or prediction is a training input.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(a):
    import lightgbm as lgb
    assert lgb.__version__ == '4.6.0'
    started = time.time()
    out = a.out.resolve()
    assert not out.exists()
    out.mkdir(parents=True)
    names = json.loads(a.names.read_text())
    with np.load(a.inputs) as z:
        d = {n: z[n] for n in ['key', 'table', 'X', 'y', 'ypair']}
    assert d['X'].shape == (223749, 345) and len(names) == 345
    tabs = np.unique(d['table'])
    assert len(tabs) == 397
    historical = np.random.default_rng(0).permutation(len(tabs)) % 5
    table_fold = (np.arange(400, dtype=np.int16) % 5).astype(np.int8)
    table_fold[tabs] = historical
    fold = table_fold[d['table']]
    keep = d['y'] | (d['ypair'] == 0)
    assert keep.sum() == 180437 and d['y'].sum() == 1817
    params = dict(objective='binary', metric='auc', learning_rate=.05, num_leaves=31,
        min_child_samples=30, feature_fraction=.35, bagging_fraction=.8,
        bagging_freq=1, lambda_l2=5., verbose=-1, n_jobs=2, max_bin=63, seed=42)
    receipt = dict(status='RUNNING', source_sha256=sha(Path(__file__)),
        inputs_sha256=sha(a.inputs), names_sha256=sha(a.names), params=params,
        rounds=400, table_fold=table_fold.tolist(), labeled_tables=tabs.tolist(),
        fold_recipe='default_rng(0).permutation(397) % 5 on sorted labeled tables; unlabeled tables use t % 5',
        models={}, quantiles={}, reads_retained_models=False, public_upload=False,
        scope='Historical asset reproduction; no new validation metric or fold search')
    target = out / 'training.json'
    target.write_text(json.dumps(receipt, indent=2))
    oof = np.full(len(fold), np.nan, dtype=np.float64)
    for f in range(5):
        va = fold == f
        tr = (~va) & keep
        assert not set(d['table'][tr]) & set(d['table'][va])
        model = lgb.train(params, lgb.Dataset(d['X'][tr], d['y'][tr].astype(int),
                          feature_name=names), num_boost_round=400)
        model_path = out / f'inst_M1_fs0_f{f}.txt'
        model.save_model(str(model_path))
        oof[va] = model.predict(d['X'][va], num_threads=2)
        neg = va & (d['ypair'] == 0)
        receipt['quantiles'][str(f)] = {str(q): float(np.quantile(oof[neg], q)) for q in [.5, .9, .99]}
        receipt['quantiles'][str(f)]['n_neg_rows'] = int(neg.sum())
        receipt['models'][str(f)] = dict(sha256=sha(model_path), training_rows=int(tr.sum()),
            validation_rows=int(va.sum()), train_tables=np.unique(d['table'][tr]).tolist(),
            valid_tables=np.unique(d['table'][va]).tolist())
        target.write_text(json.dumps(receipt, indent=2))
        print('INSTANCE TEACHER COMPLETE', f, round(time.time() - started), flush=True)
    assert np.isfinite(oof).all()
    np.save(out / 'oof.npy', oof)
    receipt.update(status='PASS', oof_sha256=sha(out / 'oof.npy'), seconds=time.time() - started,
        q_oof_mean={str(q): float(np.mean([receipt['quantiles'][str(f)][str(q)] for f in range(5)]))
                    for q in [.5, .9, .99]})
    target.write_text(json.dumps(receipt, indent=2))
    print('FIVE INSTANCE TEACHERS COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--names', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
