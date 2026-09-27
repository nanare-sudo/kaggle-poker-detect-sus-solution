"""Portable Reserve evidence from generated selection keys and current weights.

No old CSV, prediction file or evidence list is read. selection.npz is generated
by current_finish after both TabICL contexts and the old gate score exist.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V4'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
os.environ['POKER_CACHE']=str(ROOT/'cache')
os.environ['POKER_CHEN_ORDER_FILE']=str(ROOT/'pipeline/chen_order_cpu.json')
sys.path.insert(0,str(ROOT/'src'))
import reserve_features as LIVE
STATE=None


def aux(prob,key):
    p=np.clip(prob,1e-7,1-1e-7);logits=np.log(p/(1-p))
    frame=pd.DataFrame(logits);gb=frame.groupby(key)
    rank=gb.rank(pct=True).to_numpy()
    z=((frame-gb.transform('mean'))/gb.transform('std').fillna(0).clip(lower=1e-6)).clip(-8,8).to_numpy()
    return np.column_stack([logits,logits[:,1]-logits[:,0],rank,z]).astype(np.float32)


def init():
    global STATE
    import lightgbm as lgb
    import xgboost as xgb
    assert xgb.__version__=='3.2.0'
    assets=ROOT/'assets/reserve';spec=json.loads((assets/'specification.json').read_text())
    assert not spec['smoke'] and spec['inner_fold_seed']==0
    selection=np.load(ROOT/'selection.npz');keys=selection['key'];route=selection['route']
    assert pd.Index(keys).is_unique and np.isin(route,[0,1,2]).all()
    pt=np.full(12000,-1,np.int16);E=LIVE.S.engine
    for seat in range(6):pt[E.A('seat_player')[:,seat]]=E.A('hand_table')
    tables=pt[keys//12000];assert (tables>=0).all()
    tree=lambda name:lgb.Booster(model_file=str(assets/name))
    maps=[]
    for g in range(3):
        model=xgb.Booster(params={'nthread':1});model.load_model(assets/f'MAP_AUG_g{g}.ubj');maps.append(model)
    STATE=dict(spec=spec,by={t:keys[tables==t] for t in np.unique(tables)},route=dict(zip(keys,route)),
        teachers={arm:[tree(f'{arm}_inner{f}.txt') for f in range(5)] for arm in ['CLASSIC','NN_EVENTS']},
        lgb=[tree(f'LGB_AUX_g{g}.txt') for g in range(3)],maps=maps,
        columns=[json.loads((assets/f'family{g}.json').read_text()) for g in range(3)])


def work(table):
    import xgboost as xgb
    s=STATE;start=time.time()
    if table not in s['by']:return table,dict(key=np.array([],np.int64),h=np.empty((0,5),np.int64))
    data,teacher_X,teacher_names=LIVE.live(table,1,s['by'][table])
    k,h,to,base,nm,R,rn,C,cn,events,en=data
    probability=np.empty((5,len(k),2))
    for j,arm in enumerate(['CLASSIC','NN_EVENTS']):
        assert teacher_names[arm]==s['spec']['teacher_features'][arm]
        for f,model in enumerate(s['teachers'][arm]):
            probability[f,:,j]=model.predict(teacher_X[arm],num_threads=1)
    route=np.array([s['route'][q] for q in k]);scores=np.zeros(len(k))
    for g in range(3):
        take=route==g
        if not take.any():continue
        X,names=LIVE.S.G.matrix(k[take],to[take],base[take],nm,R[take],rn,C[take],cn,events[take],en,g,'CHRONO_ADD')
        assert names==s['columns'][g]['map_features']
        rank=lambda p:pd.Series(p).groupby(k[take]).rank(pct=True).to_numpy()
        values=[]
        for f in range(5):
            mat=np.concatenate([X,aux(probability[f,take],k[take])],1)
            values.append(rank(s['lgb'][g].predict(mat,num_threads=1)))
        scores[take]=.5*np.mean(values,axis=0)+.5*rank(s['maps'][g].predict(xgb.DMatrix(X,feature_names=names,nthread=1)))
    allowed=LIVE.S.AF.mask(k,h);keys,first,count=np.unique(k,return_index=True,return_counts=True)
    chosen=np.full((len(keys),5),-1,np.int64)
    for j,(a,c) in enumerate(zip(first,count)):
        ix=np.arange(a,a+c)[allowed[a:a+c]];picked=ix[np.lexsort((to[ix],-scores[ix]))[:5]]
        assert len(picked)==5;chosen[j]=h[picked]
    print('PORTABLE RESERVE',table,len(keys),round(time.time()-start),flush=True)
    return table,dict(key=keys,h=chosen)


def main(args):
    tables=list(range(400)) if args.tables=='all' else [int(t) for t in args.tables.split(',')]
    out=ROOT/'reserve_predictions';out.mkdir(exist_ok=True)
    if args.jobs==1:
        init()
        for table in tables:
            _,values=work(table);path=out/f'{table:03d}.npz';assert not path.exists();np.savez(path,**values)
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(args.jobs,initializer=init) as pool:
            for table,values in pool.imap_unordered(work,tables):
                path=out/f'{table:03d}.npz';assert not path.exists();np.savez(path,**values)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tables',default='all');parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args();assert 1<=args.jobs<=4;main(args)
