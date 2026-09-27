"""Rebuild TabICL development contexts from the isolated gameplay cache.

The fitted model's column names are fixed assets. Training rows are sampled
again from the development labels and exposure-matched unknown pairs.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['POLICY_Q2']='1';os.environ['POLICY_Q2_SFX']='_q2nnnoply'
for name in ['POLICY_FIX','POLICY_V2','SURP_STRAT','ENGINE_WK','ENGINE_SZ','ENGINE_CF','AGG_QREPLACE','AGG_EPI']:
    os.environ[name]='0'
os.environ['AGG_QFRACS']=''
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
os.environ['POKER_CACHE']=str(ROOT/'cache')


def work(table):
    import engine as E
    import aggregate as A
    import derive
    t=time.time();ht,hp,st=[E.A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    sel=np.flatnonzero((ht==table)&(hp==0));sel=sel[np.argsort(st[sel],kind='stable')]
    key,h,to,X,names=E.table_pair_hands(sel)
    u,F,fn=A.aggregate_table(key,to,X,names)
    shared=F[:,fn.index('n_shared')]
    derived,dn=derive.add_ratio_features(F,fn)
    columns=json.loads((ROOT/'assets/recipe.json').read_text())['tabicl_features']
    mat=np.nan_to_num(derived[:,[dn.index(n) for n in columns]],nan=0,posinf=0,neginf=0)
    print('CONTEXT TABLE',table,round(time.time()-t),'seconds',flush=True)
    return table,dict(key=u,X=mat,n_shared=shared)


def main(a):
    from multiprocessing import get_context
    out=ROOT/'context_replay';out.mkdir(exist_ok=True)
    todo=[t for t in range(a.tables) if not (out/f'{t:03d}.npz').exists()]
    with get_context('spawn').Pool(a.jobs) as pool:
        for t,values in pool.imap(work,todo):np.savez(out/f'{t:03d}.npz',**values)
    if a.tables!=400:return
    parts=[np.load(out/f'{t:03d}.npz') for t in range(400)]
    keys=np.concatenate([z['key'] for z in parts]);order=np.argsort(keys);keys=keys[order]
    X=np.concatenate([z['X'] for z in parts])[order]
    uni=np.concatenate([z['n_shared'] for z in parts])[order]>=57
    labels=pd.read_parquet(ROOT/'cache/labels.parquet')
    lk=labels.plo.to_numpy().astype(np.int64)*12000+labels.phi.to_numpy()
    rows=np.searchsorted(keys,lk);assert np.array_equal(keys[rows],lk)
    y=np.full(len(keys),-1,np.int8);y[rows]=labels.label.to_numpy()
    labeled=np.flatnonzero((y>=0)&uni);unknown=np.flatnonzero(uni&(y<0))
    assert len(labeled)==1860
    for j in [0,1]:
        rng=np.random.default_rng(5200+j)
        rows=np.r_[labeled,rng.choice(unknown,8000,replace=False)];rows=rows[rng.permutation(len(rows))]
        np.savez_compressed(out/f'context{j}.npz',X_train=X[rows],y_train=(y[rows]==1).astype(np.int8),key_train=keys[rows])
    print('CONTEXTS COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',type=int,default=400);p.add_argument('--jobs',type=int,default=4)
    main(p.parse_args())
