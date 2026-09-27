"""Actual S5 typed evidence from live features and retained model weights.

No historical hand score or evidence list is used by inference. --check reads
the historical table outputs only after each new table has been generated.
The parent package supplies the separately reproduced historical gate/router.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V4'
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent/'package'
sys.path[:0]=[str(PACKAGE/'pipeline'),str(HERE/'source')]
import current_reserve as CR
import tdlib
MODELS=None

def init():
    global MODELS
    import lightgbm as lgb
    CR.init()
    folder=HERE/'assets/typed'
    meta=json.loads((folder/'meta.json').read_text())
    MODELS=dict(meta=meta, models={(t,g):[
        lgb.Booster(model_file=str(folder/f'{t}_g{g}_s{s}.txt'),params={'num_threads':1})
        for s in meta['seeds']] for t in ['t1','t2'] for g in range(3)})

def work(table):
    import xgboost as xgb
    start_time=time.time();s=CR.STATE
    assert table in s['by']
    data,teacher_X,teacher_names=CR.LIVE.live(table,1,s['by'][table])
    k,h,to,base,nm,R,rn,C,cn,events,en=data
    prob=np.empty((5,len(k),2))
    for j,arm in enumerate(['CLASSIC','NN_EVENTS']):
        assert teacher_names[arm]==s['spec']['teacher_features'][arm]
        for f,m in enumerate(s['teachers'][arm]):
            prob[f,:,j]=m.predict(teacher_X[arm],num_threads=1)
    route=np.array([s['route'][q] for q in k])
    scores=np.zeros(len(k));q1=np.zeros(len(k));q2=np.zeros(len(k))
    for g in range(3):
        take=route==g
        if not take.any():continue
        X,names=CR.LIVE.S.G.matrix(k[take],to[take],base[take],nm,R[take],rn,C[take],cn,events[take],en,g,'CHRONO_ADD')
        assert names==s['columns'][g]['map_features']==MODELS['meta']['meta'][str(g)]['names']
        rank=lambda p:pd.Series(p).groupby(k[take]).rank(pct=True).to_numpy()
        values=[]
        for f in range(5):
            mat=np.concatenate([X,CR.aux(prob[f,take],k[take])],1)
            values.append(rank(s['lgb'][g].predict(mat,num_threads=1)))
        scores[take]=.5*np.mean(values,axis=0)+.5*rank(s['maps'][g].predict(xgb.DMatrix(X,feature_names=names,nthread=1)))
        q1[take]=np.mean([m.predict(X,num_threads=1) for m in MODELS['models']['t1',g]],0)
        q2[take]=np.mean([m.predict(X,num_threads=1) for m in MODELS['models']['t2',g]],0)
    allowed=CR.LIVE.S.AF.mask(k,h)
    ts=np.asarray(CR.LIVE.S.engine.A('hand_started_at')[h]).astype(np.int64)
    keys,first,count=np.unique(k,return_index=True,return_counts=True)
    assert np.all(np.diff(k)>=0)
    dec=tdlib.decode(np.clip(1.4*q1,0,1),np.clip(1.4*q2,0,1),ts,first,count)
    blended=.5*pd.Series(scores).groupby(k).rank(pct=True).to_numpy()+.5*pd.Series(dec).groupby(k).rank(pct=True).to_numpy()
    chosen=np.full((len(keys),5),-1,np.int64)
    for j,(a,c) in enumerate(zip(first,count)):
        rows=np.arange(a,a+c)[allowed[a:a+c]]
        pick=rows[np.lexsort((to[rows],-blended[rows]))[:5]]
        assert len(pick)==5;chosen[j]=h[pick]
    result=dict(k=k,h=h,to=to,ts=ts,allowed=allowed,new=scores,q1=q1,q2=q2,dec=dec,key=keys,picks=chosen)
    print('TYPED',table,len(k),len(keys),round(time.time()-start_time,2),flush=True)
    return table,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--tables',default='all');p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--check',type=Path);a=p.parse_args();assert 1<=a.jobs<=4
    tables=list(range(400)) if a.tables=='all' else list(map(int,a.tables.split(',')))
    out=HERE/'typed_tables';out.mkdir(exist_ok=True)
    todo=[t for t in tables if not (out/f'{t:03d}.npz').exists()]
    def consume(t,result):
        path=out/f'{t:03d}.npz';assert not path.exists();np.savez_compressed(path,**result)
    if a.jobs==1:
        init()
        for t in todo:consume(*work(t))
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(a.jobs,initializer=init) as pool:
            for t,result in pool.imap_unordered(work,todo):consume(t,result)
    if a.check:
        rows=pairs=0
        for t in tables:
            z=np.load(out/f'{t:03d}.npz');ref=np.load(a.check/f'{t:03d}.npz')
            for name in ref.files:np.testing.assert_array_equal(z[name],ref[name],err_msg=f'{t}:{name}')
            rows+=len(z['k']);pairs+=len(z['key'])
        report=dict(tables=tables,hand_rows=rows,pairs=pairs,all_intermediates_exact=True,
                    scope='Live retained-weight inference; historical tables read for comparison only.')
        name='TYPED_ALL.json' if len(tables)==400 else 'TYPED_CELL.json'
        (HERE/name).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
