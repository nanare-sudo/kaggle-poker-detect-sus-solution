"""Isolated inference from rebuilt gameplay arrays and fitted model assets.

Run a copy below a separate solution root containing src/, cache/ and assets/.
No original feature matrix, prediction, routing file or submission is read.
Both evidence variants are calculated for every pair; the risk gate is applied
only after the pair scores have been reconstructed.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['SEQ_FEAT']='1'
for name in ['POLICY_FIX','POLICY_V2','SURP_STRAT','ENGINE_WK','ENGINE_SZ','ENGINE_CF',
             'AGG_QREPLACE','AGG_EPI']:
    os.environ[name]='0'
os.environ['AGG_QFRACS']=''
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
os.environ['POKER_CACHE']=str(ROOT/'cache')
ASSETS=ROOT/'assets'
STATE=None


def init():
    global STATE
    import lightgbm as lgb
    import xgboost as xgb
    spec=json.loads((ASSETS/'recipe.json').read_text())
    def tree(name):return lgb.Booster(model_file=str(ASSETS/name),params={'num_threads':1})
    maps=[]
    for g in range(3):
        m=xgb.Booster(params={'nthread':1});m.load_model(ASSETS/f'map_family{g}.ubj');maps.append(m)
    STATE=dict(spec=spec,anchor=[tree(f'anchor/anchor_seed{s}.txt') for s in range(42,47)],
        behavior=tree('behavior/behavior.txt'),pool=[tree(f'pool/pool_seed{s}.txt') for s in [42,43]],
        instance=tree('instance.txt'),maps=maps,
        evidence=[[tree(f'evidence_family{g}_seed{s}.txt') for s in [42,43,44]] for g in range(3)])
    for group in ['anchor','behavior']:
        STATE[group+'_keep']=np.load(ASSETS/group/'base_keep.npy')
        STATE[group+'_names']=json.loads((ASSETS/group/'feature_names.json').read_text())
    ep=pd.read_parquet(ROOT/'cache/eval_pairs.parquet')
    STATE['want']=np.sort(ep.plo.to_numpy().astype(np.int64)*12000+ep.phi.to_numpy())


def reduced(F,names,group):
    import derive
    keep=STATE[group+'_keep']
    X,nm=derive.add_ratio_features(F[:,keep],[names[i] for i in keep])
    assert nm==STATE[group+'_names']
    return X


def work(table):
    import engine as E
    import aggregate as A
    import derive
    import evidence as EV
    import seqwire
    import pool_inference as B3
    import hand_features as S
    import xgboost as xgb
    t=time.time();ht,hp,st=[E.A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    sel=np.flatnonzero((ht==table)&(hp==1));sel=sel[np.argsort(st[sel],kind='stable')]
    # All partners at the table contribute to opponent-relative baselines.
    E.POLICY_Q2=True;E.POLICY_Q2_SFX='_q2nnnoply'
    k,h,to,X,names=E.table_pair_hands(sel)
    u,F,fn=A.aggregate_table(k,to,X,names)
    take=np.isin(u,STATE['want']);keys=u[take];F=F[take]
    anchorX=reduced(F,fn,'anchor')
    anchor=np.column_stack([m.predict(anchorX,num_threads=1) for m in STATE['anchor']])
    allF,allnames=derive.add_ratio_features(F,fn)
    tabicl=np.nan_to_num(allF[:,[allnames.index(n) for n in STATE['spec']['tabicl_features']]],nan=0,posinf=0,neginf=0)
    del X,F,allF,anchorX
    E.POLICY_Q2=False
    k,h,to,X,names=E.table_pair_hands(sel)
    u,F,fn=A.aggregate_table(k,to,X,names)
    assert np.array_equal(u[take],keys)
    behavior=STATE['behavior'].predict(reduced(F[take],fn,'behavior'),num_threads=1)
    del F
    take=np.isin(k,keys);k,h,to,X=k[take],h[take],to[take],X[take]
    order=np.lexsort((to,k));k,h,to,X=k[order],h[order],to[order],X[order]
    seq,_=seqwire.live(sel,k,h)
    instance=STATE['instance'].predict(np.concatenate([X[:,EV.evid_keep(names)],seq],axis=1),num_threads=1)
    u,start,count=np.unique(k,return_index=True,return_counts=True);assert np.array_equal(u,keys)
    threshold=STATE['spec']['pool_thresholds']
    poolX=B3._agg(u,start,count,instance,to.astype(np.float32)/max(len(sel)-1,1),threshold['0.9'],threshold['0.99'])
    pool=np.column_stack([m.predict(poolX,num_threads=1) for m in STATE['pool']])
    del X,seq,instance
    E.POLICY_Q2=True;E.POLICY_Q2_SFX='_q2'
    k,h,to,X,names,R,rn,C,cn,events,en=S.live_data(table,1,keys)
    u,start,count=np.unique(k,return_index=True,return_counts=True);assert np.array_equal(u,keys)
    route=np.repeat(behavior.argmax(1),count);scores=np.zeros((len(k),2))
    for g in range(3):
        take=route==g
        if not take.any():continue
        mat,nm=S.G.matrix(k[take],to[take],X[take],names,R[take],rn,C[take],cn,events[take],en,g,'CHRONO_ADD')
        assert nm==STATE['spec']['evidence_features'][str(g)]
        rank=lambda v:pd.Series(v).groupby(k[take]).rank(pct=True).to_numpy()
        a=np.mean([rank(m.predict(mat,num_threads=1)) for m in STATE['evidence'][g]],axis=0)
        b=rank(STATE['maps'][g].predict(xgb.DMatrix(mat,feature_names=nm,nthread=1)))
        scores[take,0]=a;scores[take,1]=.5*a+.5*b
    allowed=S.AF.mask(k,h);chosen=np.full((len(u),2,5),-1,np.int64)
    for j,(a,c) in enumerate(zip(start,count)):
        ix=np.arange(a,a+c)[allowed[a:a+c]]
        for variant in range(2):
            hs=h[ix[np.lexsort((to[ix],-scores[ix,variant]))[:5]]]
            chosen[j,variant,:len(hs)]=hs
    print('REPLAY',table,len(keys),round(time.time()-t),'seconds',flush=True)
    return table,dict(key=keys,anchor=anchor,behavior=behavior,pool=pool,tabicl=tabicl,evidence=chosen)


def main(a):
    required=['act_equity','expect_table']+[n+s for n in ['act_surprise','act_paggr','act_pfold'] for s in ['','_q2','_q2nnnoply']]
    for n in required:assert (ROOT/f'cache/{n}.npy').exists(),n
    import evidence_filter as AF
    AF.build_acted()  # One writer before spawning workers.
    out=ROOT/'out';out.mkdir(exist_ok=True)
    files=list((ROOT/'src').glob('*.py'))+[p for p in (ROOT/'assets').rglob('*') if p.is_file()]
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checksum=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    (out/'source_manifest.json').write_text(json.dumps(frozen,indent=2))
    todo=[]
    for table in range(a.tables):
        path=out/f'{table:03d}.npz'
        if path.exists():assert str(np.load(path)['manifest_sha256'])==checksum
        else:todo.append(table)
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.jobs,initializer=init) as pool:
        for table,values in pool.imap(work,todo):
            np.savez(out/f'{table:03d}.npz',**values,manifest_sha256=checksum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',type=int,default=400);p.add_argument('--jobs',type=int,default=2)
    main(p.parse_args())
