"""Fit both original pool heads from newly rebuilt universe aggregates."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    import lightgbm as lgb
    assert lgb.__version__ == '4.6.0'
    started = time.time()
    root, out = a.root.resolve(), a.out.resolve()
    assert not out.exists()
    out.mkdir(parents=True)
    sys.path.insert(0, str(root / 'src'))
    import model as M
    spec = json.loads((a.inputs / 'specification.json').read_text())
    names = json.loads((a.inputs / 'feature_names.json').read_text())
    chunks = []
    for t in range(400):
        with np.load(a.inputs / f'{t:03d}.npz') as z:
            assert str(z['source_sha256']) == spec['source_sha256']
            assert (z['table'] == t).all()
            chunks.append({n: z[n] for n in ['key', 'X', 'table']})
    merged = {n: np.concatenate([d[n] for d in chunks]) for n in chunks[0]}
    order = np.argsort(merged['key'], kind='stable')
    data = {n: v[order] for n, v in merged.items()}
    assert data['X'].shape == (137739, 18) and len(names) == 18
    assert (np.diff(data['key']) > 0).all() and np.isfinite(data['X']).all()
    labels = pd.read_parquet(root / 'cache/labels.parquet')
    label_keys = labels.plo.to_numpy().astype(np.int64) * 12000 + labels.phi.to_numpy()
    at = np.searchsorted(data['key'], label_keys)
    np.testing.assert_array_equal(data['key'][at], label_keys)
    y = np.full(len(data['key']), -1, dtype=np.int8)
    y[at] = labels.label.to_numpy()
    assert int((y == 1).sum()) == 372 and int((y >= 0).sum()) == 1860
    data['label'] = y
    np.savez_compressed(out / 'training_inputs.npz', **data)
    report = dict(status='RUNNING', source_sha256=sha(Path(__file__)),
        parameter_source_sha256=sha(root / 'src/model.py'),
        feature_source_sha256=spec['source_sha256'],
        training_inputs_sha256=sha(out / 'training_inputs.npz'),
        rows=len(y), known_positive=int((y == 1).sum()), known_negative=int((y == 0).sum()),
        unlabeled=int((y < 0).sum()), feature_names=names, rounds=400, models={},
        label_semantics='Historical positive-unlabeled fit: unlisted pairs receive a zero training target, not a confirmed-negative ground-truth claim',
        reads_retained_models=False, public_upload=False)
    for seed in [42, 43]:
        params = dict(M.PARAMS)
        params.update(n_jobs=2, seed=seed)
        model = lgb.train(params, lgb.Dataset(data['X'], (y == 1).astype(int), feature_name=names),
                          num_boost_round=400)
        path = out / f'pool_seed{seed}.txt'
        model.save_model(str(path))
        report['models'][str(seed)] = dict(sha256=sha(path), params=params)
        (out / 'training.json').write_text(json.dumps(report, indent=2))
        print('POOL HEAD COMPLETE', seed, round(time.time() - started), flush=True)
    report.update(status='PASS', seconds=time.time() - started)
    (out / 'training.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
