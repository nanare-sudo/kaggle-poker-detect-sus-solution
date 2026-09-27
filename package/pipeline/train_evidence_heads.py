"""Fit the deployed nine LGB and three MAP heads from rebuilt raw matrices."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


MAP_PARAMS = dict(objective='rank:map', tree_method='hist', eta=.05, max_depth=6,
    min_child_weight=5, reg_lambda=5., subsample=.8, colsample_bytree=.35,
    seed=42, nthread=2, lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=8,
    eval_metric='map@5')


def main(a):
    import lightgbm as lgb
    import xgboost as xgb
    assert lgb.__version__ == '4.6.0' and xgb.__version__ == '3.2.0'
    started = time.time()
    root, out = a.root.resolve(), a.out.resolve()
    assert not out.exists()
    out.mkdir(parents=True)
    sys.path.insert(0, str(root / 'src'))
    import evidence as EV
    names = json.loads((a.inputs / 'feature_names.json').read_text())
    assert sorted(names) == ['0', '1', '2']
    spec = json.loads((a.inputs / 'specification.json').read_text())
    data = {g: [] for g in range(3)}
    for t in range(400):
        with np.load(a.inputs / f'{t:03d}.npz') as z:
            assert str(z['source_sha256']) == spec['source_sha256']
            for g in range(3):
                if f'g{g}_key' in z:
                    data[g].append({n: z[f'g{g}_{n}'] for n in ['key', 'h', 'torder', 'y', 'X']})
    report = dict(status='RUNNING', source_sha256=sha(Path(__file__)),
        parameter_source_sha256=sha(root / 'src/evidence.py'),
        feature_source_sha256=spec['source_sha256'], rounds=750,
        lightgbm=lgb.__version__, xgboost=xgb.__version__, numpy=np.__version__,
        map_params=MAP_PARAMS, families={}, reads_retained_models=False, public_upload=False,
        scope='Original final evidence heads; original rebuilt action references, no all-component fresh-neural claim')
    rows, positives = 0, 0
    for g in range(3):
        merged = {n: np.concatenate([d[n] for d in data[g]]) for n in data[g][0]}
        order = np.lexsort((merged['torder'], merged['key']))
        d = {n: v[order] for n, v in merged.items()}
        data[g] = []
        assert d['X'].shape[1] == len(names[str(g)]) and np.isfinite(d['X']).all()
        rows += len(d['key'])
        positives += int(d['y'].sum())
        path = out / f'family{g}_training.npz'
        np.savez_compressed(path, **d)
        family = dict(rows=len(d['key']), pairs=len(np.unique(d['key'])), positives=int(d['y'].sum()),
            columns=len(names[str(g)]), training_inputs_sha256=sha(path), models={})
        groups = np.unique(d['key'], return_counts=True)[1]
        dataset = lgb.Dataset(d['X'], d['y'], group=groups, feature_name=names[str(g)])
        for seed in [42, 43, 44]:
            params = dict(EV.RANK_PARAMS)
            params.update(seed=seed, n_jobs=1, deterministic=True, force_col_wise=True)
            model = lgb.train(params, dataset, num_boost_round=750)
            path = out / f'evidence_family{g}_seed{seed}.txt'
            model.save_model(str(path))
            family['models'][path.name] = dict(sha256=sha(path), params=params)
            print('FINAL EVIDENCE LGB', g, seed, round(time.time() - started), flush=True)
        matrix = xgb.DMatrix(d['X'], label=d['y'], group=groups, feature_names=names[str(g)], nthread=2)
        model = xgb.train(MAP_PARAMS, matrix, num_boost_round=750)
        path = out / f'map_family{g}.ubj'
        model.save_model(path)
        family['models'][path.name] = dict(sha256=sha(path), params=MAP_PARAMS)
        report['families'][str(g)] = family
        (out / 'training.json').write_text(json.dumps(report, indent=2))
        print('FINAL EVIDENCE MAP', g, round(time.time() - started), flush=True)
    assert rows == 45129 and positives == 1817
    report.update(status='PASS', rows=rows, positives=positives, seconds=time.time() - started)
    (out / 'training.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
