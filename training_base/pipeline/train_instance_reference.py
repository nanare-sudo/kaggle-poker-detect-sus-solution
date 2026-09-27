"""Train the original final instance model from regenerated labeled-pair hands.

Uses only listed evidence as positive and confirmed-negative pair hands as
negative. Unlisted hands of positive pairs are excluded from this binary fit.
This component does not regenerate the folded pool inputs or pool heads.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def main(a):
    import lightgbm as lgb
    assert lgb.__version__=='4.6.0'
    started=time.time();inputs=a.inputs.resolve();out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    spec=json.loads((inputs/'specification.json').read_text());names=json.loads((inputs/'feature_names.json').read_text())
    assert len(names)==345
    chunks=[]
    for t in range(400):
        with np.load(inputs/f'{t:03d}.npz') as z:
            assert str(z['source_sha256'])==spec['source_sha256']
            chunks.append({n:z[n] for n in ['key','h','torder','X','y','ypair','table']})
    merged={n:np.concatenate([d[n] for d in chunks]) for n in chunks[0]};del chunks
    order=np.lexsort((merged['torder'],merged['key']));d={n:v[order] for n,v in merged.items()};del merged
    assert len(d['key'])==223749 and d['X'].shape==(223749,345)
    keep=d['y']|(d['ypair']==0);assert int(keep.sum())==180437 and int(d['y'].sum())==1817
    np.savez_compressed(out/'training_inputs.npz',**d)
    params=dict(objective='binary',metric='auc',learning_rate=.05,num_leaves=31,
        min_child_samples=30,feature_fraction=.35,bagging_fraction=.8,bagging_freq=1,
        lambda_l2=5.,verbose=-1,n_jobs=2,max_bin=63,seed=42)
    model=lgb.train(params,lgb.Dataset(d['X'][keep],d['y'][keep].astype(int),feature_name=names),num_boost_round=400)
    target=out/'instance.txt';model.save_model(str(target))
    report=dict(status='PASS',source_sha256=sha(Path(__file__)),feature_source_sha256=spec['source_sha256'],
        raw_feature_rows=len(d['key']),training_rows=int(keep.sum()),positive_rows=int(d['y'].sum()),
        columns=names,params=params,rounds=400,model_sha256=sha(target),
        training_inputs_sha256=sha(out/'training_inputs.npz'),seconds=time.time()-started,
        reads_retained_weights=False,public_upload=False,
        scope='Final instance model only; original reconstructed action references, no pool-fold reproduction claim')
    (out/'training.json').write_text(json.dumps(report,indent=2));print('FINAL INSTANCE TRAINING COMPLETE',report['model_sha256'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
