"""Train original anchor/behavior heads from freshly rebuilt pair aggregates.

Re-runs the supervised feature selectors on all final-training data. Retained
feature masks, fitted models and evaluation predictions are never inputs.
"""
import os
os.environ['OMP_NUM_THREADS']='3';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def assemble(root,inputs,tag):
    specification=json.loads((inputs/'specification.json').read_text())
    paths=[inputs/f'{i:03d}.npz' for i in range(400)]
    keys=[]
    for p in paths:
        with np.load(p) as z:
            assert str(z['source_sha256'])==specification['source_sha256']
            keys.append(z[tag+'_key'])
    key=np.concatenate(keys);order=np.argsort(key,kind='stable')
    assert len(np.unique(key))==len(key)
    dest=np.empty(len(key),np.int64);dest[order]=np.arange(len(key))
    names=json.loads((inputs/'feature_names.json').read_text())[tag]
    folder=root/'features';folder.mkdir(exist_ok=True)
    target=folder/f'pair_X_{tag}.npy';part=folder/f'pair_X_{tag}.part.npy'
    assert not target.exists() and not part.exists()
    matrix=np.lib.format.open_memmap(part,mode='w+',dtype=np.float32,shape=(len(key),len(names)))
    start=0
    for p,k in zip(paths,keys):
        with np.load(p) as z:
            x=z[tag+'_X'];assert x.shape==(len(k),len(names))
            matrix[dest[start:start+len(k)]]=x
        start+=len(k)
    assert start==len(key)
    matrix.flush();del matrix;part.rename(target)
    np.save(folder/f'pair_key_{tag}.npy',key[order])
    (folder/f'pair_names_{tag}.json').write_text(json.dumps(names))
    return dict(rows=len(key),columns=len(names),sha256=sha(target),
        key_sha256=sha(folder/f'pair_key_{tag}.npy'),input_source_sha256=specification['source_sha256'])


def train(root,out,kind):
    import lightgbm as lgb
    import feature_selection as MS
    import model as M
    import behavior as B
    assert lgb.__version__=='4.6.0'
    tag='dev_q2nnnoply' if kind=='anchor' else 'dev_w2000'
    k,y,fam,tab,uni,raw,keep,names=MS._select_columns(1200,tag=tag)
    X=MS._materialize(tag,keep,raw,names)
    target=out/kind;assert not target.exists();target.mkdir()
    np.save(target/'base_keep.npy',keep)
    (target/'feature_names.json').write_text(json.dumps(names))
    reports=[]
    if kind=='anchor':
        train_rows=np.flatnonzero(uni);truth=(y[train_rows]==1).astype(int)
        for seed in range(42,47):
            params=dict(M.PARAMS,seed=seed,n_jobs=3)
            m=lgb.train(params,lgb.Dataset(X[train_rows],truth,feature_name=names),num_boost_round=500)
            path=target/f'anchor_seed{seed}.txt';m.save_model(str(path))
            reports.append(dict(seed=seed,rounds=500,rows=len(train_rows),sha256=sha(path)))
            print('ANCHOR TRAINED',reports[-1],flush=True)
    else:
        m=B.full_fit(X,fam,names,n_rounds=600)
        path=target/'behavior.txt';m.save_model(str(path))
        reports.append(dict(seed=42,rounds=600,rows=int(np.isin(fam,B.FAMS).sum()),sha256=sha(path)))
        print('BEHAVIOR TRAINED',reports[-1],flush=True)
    return dict(models=reports,columns=len(names),mask_sha256=sha(target/'base_keep.npy'),
        names_sha256=sha(target/'feature_names.json'),lightgbm=lgb.__version__)


def main(a):
    root=a.root.resolve();inputs=a.inputs.resolve();out=a.out.resolve()
    sys.path.insert(0,str(root/'src'))
    assert not out.exists();out.mkdir(parents=True)
    report=dict(scope='Fresh raw-derived pair aggregates, selectors and anchor/behavior fits only; no retained weights',
        driver_sha256=sha(Path(__file__)),stages={},public_upload=False)
    started=time.time()
    for kind,tag in [('anchor','dev_q2nnnoply'),('behavior','dev_w2000')]:
        report['stages'][kind]=dict(assembly=assemble(root,inputs,tag))
        (out/'training.json').write_text(json.dumps(report,indent=2))
        report['stages'][kind]['training']=train(root,out,kind)
        report['seconds']=time.time()-started
        (out/'training.json').write_text(json.dumps(report,indent=2))
    report['complete']=True
    (out/'training.json').write_text(json.dumps(report,indent=2))
    print('RAW-DERIVED PAIR HEAD TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
