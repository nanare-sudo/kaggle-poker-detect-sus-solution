"""Train Reserve evidence heads and cross-fitted auxiliary teachers."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

AUX_NAMES = ['aux_classic_logit', 'aux_nn_events_logit', 'aux_logit_difference',
             'aux_classic_pair_rank', 'aux_nn_events_pair_rank', 'aux_classic_pair_z', 'aux_nn_events_pair_z']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def aux(prob, key):
    p = np.clip(prob, 1e-7, 1-1e-7)
    logits = np.log(p/(1-p))
    frame = pd.DataFrame(logits); gb = frame.groupby(key)
    rank = gb.rank(pct=True).to_numpy()
    z = ((frame-gb.transform('mean'))/gb.transform('std').fillna(0).clip(lower=1e-6)).clip(-8,8).to_numpy()
    return np.column_stack([logits, logits[:,1]-logits[:,0], rank, z]).astype(np.float32)


def teacher(task):
    import lightgbm as lgb
    arm, fold, nested, out, smoke = task
    nested = Path(nested); out = Path(out); t = time.time()
    z = np.load(nested / 'nested_training.npz')
    meta = json.loads((nested / 'nested_experiment.json').read_text())
    use = (z['outer'] != fold) & z['eligible']  # Original outer seed zero becomes final inner exclusion.
    if smoke:
        rows = np.flatnonzero(use & (z['y'] == 0))
        sampled = np.random.default_rng(42).choice(rows, min(5000,len(rows)), replace=False)
        use = use & (z['y'] == 1)
        use[sampled] = True
    train_tables = np.unique(z['table'][use])
    assert not set(train_tables) & set(np.flatnonzero(np.array(meta['table_outer']) == fold))
    X = z[arm]
    model = lgb.train(meta['instance_params'], lgb.Dataset(X[use], z['y'][use],
        feature_name=meta['features'][arm]), num_boost_round=5 if smoke else 400)
    path = out / f'{arm}_inner{fold}.txt'; model.save_model(str(path))
    report = dict(arm=arm, inner_fold_seed=0, inner_fold=fold, rows=int(use.sum()),
        train_tables=train_tables.tolist(), model_sha256=sha(path), seconds=time.time()-t,
        smoke=smoke, lightgbm=lgb.__version__, numpy=np.__version__)
    (out / f'{arm}_inner{fold}.json').write_text(json.dumps(report, indent=2))
    print('FINAL TEACHER', arm, fold, round(time.time()-t), flush=True)


def training_aux(nested, rankdata, out):
    import lightgbm as lgb
    z = np.load(nested / 'nested_training.npz'); d = np.load(rankdata / 'identity.npz')
    code = z['key']*2000000+z['h']; wanted = d['key']*2000000+d['h']
    order = np.argsort(code); ix = order[np.searchsorted(code[order], wanted)]
    np.testing.assert_array_equal(code[ix], wanted)
    prob = np.full((len(ix),2), np.nan)
    for j, arm in enumerate(['CLASSIC', 'NN_EVENTS']):
        X = z[arm][ix]
        for fold in range(5):
            report = json.loads((out / f'{arm}_inner{fold}.json').read_text())
            assert sha(out / f'{arm}_inner{fold}.txt') == report['model_sha256']
            take = z['outer'][ix] == fold
            assert not set(report['train_tables']) & set(d['table'][take])
            m = lgb.Booster(model_file=str(out / f'{arm}_inner{fold}.txt'))
            prob[take,j] = m.predict(X[take], num_threads=1)
    assert np.isfinite(prob).all()
    np.savez(out / 'training_aux.npz', key=d['key'], h=d['h'], probability=prob)
    print('FINAL CROSS-FITTED AUXILIARIES COMPLETE', flush=True)


def ranker(task):
    import lightgbm as lgb
    import xgboost as xgb
    assert xgb.__version__ == '3.2.0'
    g, rankdata, windowdata, out, smoke = task
    rankdata = Path(rankdata); windowdata = Path(windowdata); out = Path(out); t = time.time()
    d = np.load(rankdata / 'identity.npz'); z = np.load(rankdata / f'family{g}.npz')
    names = json.loads((rankdata / 'evidence_experiment.json').read_text())['features'][str(g)]
    a = np.load(out / 'training_aux.npz')
    np.testing.assert_array_equal(a['key'], d['key']); np.testing.assert_array_equal(a['h'], d['h'])
    take = np.flatnonzero(z['train_family'])
    if smoke:
        keys = d['key'][z['row'][take]]
        take = take[np.isin(keys, np.unique(keys)[:8])]
    rows = z['row'][take]; keys = d['key'][rows]
    X = np.concatenate([z['X'][take], aux(a['probability'][rows], keys)], 1)
    lp = dict(objective='lambdarank', metric='map', eval_at=[5], lambdarank_truncation_level=30,
        learning_rate=.05, num_leaves=31, min_child_samples=30, feature_fraction=.35,
        bagging_fraction=.8, bagging_freq=1, lambda_l2=5., verbosity=-1, num_threads=1,
        seed=42, deterministic=True, force_col_wise=True)
    m = lgb.train(lp, lgb.Dataset(X, z['y'][take], group=np.unique(keys,return_counts=True)[1],
        feature_name=names+AUX_NAMES), num_boost_round=5 if smoke else 750)
    m.save_model(str(out / f'LGB_AUX_g{g}.txt'))
    p = m.predict(X[:100], num_threads=1)
    np.testing.assert_array_equal(p, lgb.Booster(model_file=str(out / f'LGB_AUX_g{g}.txt')).predict(X[:100], num_threads=1))
    del X, m
    w = np.load(windowdata / f'family{g}.npz')
    wm = json.loads((windowdata / 'window_experiment.json').read_text())
    assert wm['features'][str(g)] == names
    take = np.arange(len(w['query']))
    if smoke: take = take[np.isin(w['query'],np.unique(w['query'])[:8])]
    q = w['query'][take]
    assert (np.diff(q) >= 0).all()
    _, start, count = np.unique(q, return_index=True, return_counts=True)
    weight = w['weight'][take]
    np.testing.assert_array_equal(weight, np.repeat(weight[start], count))
    xp = dict(objective='rank:map', tree_method='hist', eta=.05, max_depth=6, min_child_weight=5,
        reg_lambda=5., subsample=.8, colsample_bytree=.35, seed=42, nthread=1,
        lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=8, eval_metric='map@5')
    m = xgb.train(xp, xgb.DMatrix(w['X'][take], label=w['y'][take], group=count,
        weight=weight[start], feature_names=names, nthread=1), num_boost_round=5 if smoke else 750)
    m.save_model(out / f'MAP_AUG_g{g}.ubj')
    (out / f'family{g}.json').write_text(json.dumps(dict(family=g, lgb_features=names+AUX_NAMES,
        map_features=names, lgb_params=lp, map_params=xp, seconds=time.time()-t,
        smoke=smoke, lightgbm=lgb.__version__, xgboost=xgb.__version__, numpy=np.__version__), indent=2))
    print('FINAL RANKERS', g, round(time.time()-t), flush=True)


def main(a):
    nested = a.nested or next(Path('/kaggle/input').rglob('nested_training.npz')).parent
    rankdata = a.rankdata or next(Path('/kaggle/input').rglob('evidence_experiment.json')).parent
    windowdata = a.windowdata or next(Path('/kaggle/input').rglob('window_experiment.json')).parent
    for folder, name in [(rankdata,'evidence_experiment.json'),(windowdata,'window_experiment.json')]:
        meta = json.loads((folder / name).read_text())
        for n, value in meta['files'].items(): assert sha(folder / n) == value
    for n, value in json.loads((nested / 'input_hashes.json').read_text()).items(): assert sha(nested / n) == value
    a.out.mkdir(parents=True, exist_ok=True)
    assert not list(a.out.glob('*.txt'))
    specification = dict(recipe='50% LGB_AUX rank +50% MAP_AUG rank', inner_fold_seed=0,
        teacher_rounds=400, ranking_rounds=750, teacher_arms=['CLASSIC','NN_EVENTS'],
        teacher_params=json.loads((nested/'nested_experiment.json').read_text())['instance_params'],
        teacher_features=json.loads((nested/'nested_experiment.json').read_text())['features'],
        source_sha256=sha(__file__), smoke=a.smoke, public_upload=False,
        status='Reserve evidence models',
        serving='Five teacher-pair variants, mean within-pair LGB_AUX ranks, 50/50 with MAP_AUG; existing action filter and top10000 risk gate',
        limitation='Final teachers train on 80% of tables versus 64% in nested CV; same development labels and adaptive recipe choice')
    (a.out / 'specification.json').write_text(json.dumps(specification, indent=2))
    tasks = [(arm, f, str(nested), str(a.out), a.smoke) for arm in ['CLASSIC','NN_EVENTS'] for f in range(5)]
    from multiprocessing import get_context
    with get_context('spawn').Pool(2 if a.smoke else 3) as pool: list(pool.imap_unordered(teacher,tasks))
    training_aux(nested,rankdata,a.out)
    tasks = [(g,str(rankdata),str(windowdata),str(a.out),a.smoke) for g in range(3)]
    with get_context('spawn').Pool(2 if a.smoke else 3) as pool: list(pool.imap_unordered(ranker,tasks))
    files = {p.name:sha(p) for p in a.out.iterdir() if p.suffix in ['.txt','.ubj','.json']}
    (a.out / 'model_manifest.json').write_text(json.dumps(files,indent=2))
    print('RESERVE MODELS COMPLETE',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ['nested','rankdata','windowdata']: p.add_argument('--'+n,type=Path)
    p.add_argument('--out',type=Path,default=Path('/kaggle/working/out'))
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
